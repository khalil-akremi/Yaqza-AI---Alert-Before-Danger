"""
Session Logger - Persistence et KPI pour sessions de conduite
Export CSV unifié fatigue + télémétrie + événements
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np


class SessionLogger:
    """Gestion de la persistence des données de session"""
    
    def __init__(self, session_id=None, output_dir='sessions'):
        """
        Args:
            session_id: ID unique de session (auto-généré si None)
            output_dir: Dossier de sortie pour les fichiers
        """
        if session_id is None:
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        self.session_id = session_id
        self.output_dir = output_dir
        
        # Créer le dossier si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        # Chemins des fichiers
        self.csv_path = os.path.join(output_dir, f'session_{session_id}.csv')
        self.summary_path = os.path.join(output_dir, f'session_{session_id}_summary.json')
        
        # Buffer de données
        self.data_buffer = []
        self.events = []
        
        # Métriques de session
        self.start_time = datetime.now()
        self.datapoint_count = 0
        
        # Ouvrir CSV et écrire header
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = None  # Initialisé au premier datapoint
        
    def log_datapoint(self, fatigue_metrics: Dict, driving_data: Dict, analysis: Dict):
        """
        Log un point de données unifié (fatigue + conduite + analyse)
        
        Args:
            fatigue_metrics: Dict avec eye_ratio, yawn_count, fatigue_score, etc.
            driving_data: Dict avec speed, acceleration, steering, etc.
            analysis: Dict avec events, eco_score, etc.
        """
        # Fusionner tous les dicts
        datapoint = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': self.session_id,
            
            # Fatigue
            'fatigue_score': fatigue_metrics.get('composite_score', 0),
            'eye_ratio': fatigue_metrics.get('eye_ratio', 0),
            'yawn_count': fatigue_metrics.get('yawn_count', 0),
            'head_pitch': fatigue_metrics.get('head_pitch', 0),
            'head_yaw': fatigue_metrics.get('head_yaw', 0),
            'head_roll': fatigue_metrics.get('head_roll', 0),
            
            # Conduite
            'speed_kmh': driving_data.get('speed_kmh', 0),
            'acceleration_ms2': driving_data.get('acceleration_ms2', 0),
            'steering_angle_deg': driving_data.get('steering_angle_deg', 0),
            'throttle_position_pct': driving_data.get('throttle_position_pct', 0),
            'brake_pressure_pct': driving_data.get('brake_pressure_pct', 0),
            'co2_estimate_gkm': driving_data.get('co2_estimate_gkm', 0),
            
            # Analyse
            'events_detected': ','.join(analysis.get('events', [])) or 'none',
            'event_count': len(analysis.get('events', [])),
            'cumulative_distance_km': analysis.get('total_distance_km', 0),
            'eco_score': analysis.get('eco_score', 100),
            
            # Recommandations (si présentes)
            'recommendation': analysis.get('recommendation', '')
        }
        
        # Initialiser CSV writer au premier datapoint
        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=datapoint.keys())
            self.csv_writer.writeheader()
        
        # Écrire dans CSV (streaming)
        self.csv_writer.writerow(datapoint)
        self.csv_file.flush()  # Force écriture immédiate
        
        # Garder en buffer pour analyse
        self.data_buffer.append(datapoint)
        self.datapoint_count += 1
        
        # Tracker événements significatifs
        if analysis.get('events'):
            for event in analysis['events']:
                if event in ['harsh_acceleration', 'harsh_braking', 'speeding']:
                    self.events.append({
                        'timestamp': datapoint['timestamp'],
                        'type': event,
                        'speed': datapoint['speed_kmh'],
                        'fatigue_score': datapoint['fatigue_score']
                    })
    
    def calculate_kpis(self) -> Dict:
        """Calcule les KPI de la session"""
        if not self.data_buffer:
            return {}
            
        df = pd.DataFrame(self.data_buffer)
        
        duration_seconds = (datetime.now() - self.start_time).total_seconds()
        duration_minutes = duration_seconds / 60
        
        # KPI Fatigue
        fatigue_scores = df['fatigue_score'].values
        max_fatigue = float(fatigue_scores.max())
        avg_fatigue = float(fatigue_scores.mean())
        
        # Temps avant fatigue significative (score > 40)
        fatigue_onset_idx = next((i for i, score in enumerate(fatigue_scores) if score > 40), len(fatigue_scores))
        time_to_fatigue_min = (fatigue_onset_idx / max(len(fatigue_scores), 1)) * duration_minutes
        
        # KPI Conduite
        total_distance = float(df['cumulative_distance_km'].iloc[-1]) if len(df) > 0 else 0
        avg_speed = float(df['speed_kmh'].mean())
        final_eco_score = float(df['eco_score'].iloc[-1]) if len(df) > 0 else 100
        
        # CO2
        avg_co2 = float(df['co2_estimate_gkm'].mean())
        optimal_co2 = 110
        co2_excess_pct = ((avg_co2 - optimal_co2) / optimal_co2) * 100
        total_co2_g = avg_co2 * total_distance * 1000
        potential_savings_g = max(0, total_co2_g - (optimal_co2 * total_distance * 1000))
        
        # Événements
        event_counts = {}
        for event_str in df['events_detected']:
            if event_str and event_str != 'none':
                for event in event_str.split(','):
                    event_counts[event] = event_counts.get(event, 0) + 1
        
        total_events = sum(event_counts.values())
        events_per_km = total_events / max(total_distance, 0.001)
        
        return {
            'duration_minutes': round(duration_minutes, 1),
            'total_datapoints': self.datapoint_count,
            'distance_km': round(total_distance, 2),
            'avg_speed_kmh': round(avg_speed, 1),
            'max_fatigue_score': round(max_fatigue, 1),
            'avg_fatigue_score': round(avg_fatigue, 1),
            'time_to_fatigue_min': round(time_to_fatigue_min, 1),
            'total_yawns': int(df['yawn_count'].sum()),
            'eco_score': round(final_eco_score, 1),
            'avg_co2_gkm': round(avg_co2, 1),
            'co2_excess_pct': round(co2_excess_pct, 1),
            'potential_co2_savings_g': round(potential_savings_g, 1),
            'total_events': total_events,
            'events_per_km': round(events_per_km, 2),
            'event_breakdown': event_counts
        }
    
    def save_summary(self) -> Dict:
        """Sauvegarde le résumé JSON de la session"""
        kpis = self.calculate_kpis()
        
        summary = {
            'session_id': self.session_id,
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'kpis': kpis,
            'events': self.events[:50],  # First 50 events
            'recommendations': self.generate_recommendations(kpis)
        }
        
        # Convert numpy types to native Python types for JSON serialization
        summary = self._convert_numpy_types(summary)
        
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        return summary
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types recursively"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def generate_recommendations(self, kpis: Dict) -> List[str]:
        """Génère des recommandations basées sur les KPI"""
        recommendations = []
        
        # Fatigue
        if kpis.get('max_fatigue_score', 0) > 70:
            recommendations.append(
                "⚠️ FATIGUE CRITIQUE: Pause immédiate recommandée! "
                "La fatigue augmente drastiquement le risque d'accident."
            )
        elif kpis.get('max_fatigue_score', 0) > 40:
            recommendations.append(
                "⚠️ FATIGUE: Planifiez une pause dans les 15 prochaines minutes."
            )
            
        if kpis.get('time_to_fatigue_min', 999) < 30:
            recommendations.append(
                f"⚠️ FATIGUE: Vous montrez des signes de fatigue après seulement "
                f"{int(kpis['time_to_fatigue_min'])} minutes. "
                f"Considérez des pauses plus fréquentes (toutes les 2h recommandé)."
            )
        
        # Éco-conduite
        event_breakdown = kpis.get('event_breakdown', {})
        
        harsh_accel = event_breakdown.get('harsh_acceleration', 0)
        if harsh_accel > 0:
            recommendations.append(
                f"🚗 CONDUITE: {harsh_accel} accélérations brusques détectées. "
                f"Des accélérations progressives réduiraient votre consommation de 10-15%."
            )
            
        harsh_brake = event_breakdown.get('harsh_braking', 0)
        if harsh_brake > 0:
            recommendations.append(
                f"🚗 CONDUITE: {harsh_brake} freinages brusques. "
                f"Anticipez le trafic pour freiner en douceur et récupérer de l'énergie."
            )
        
        if kpis.get('co2_excess_pct', 0) > 10:
            recommendations.append(
                f"🌱 ÉCO-IMPACT: Votre consommation est {kpis['co2_excess_pct']:.1f}% "
                f"au-dessus de l'optimal. Potentiel d'économie: {kpis['potential_co2_savings_g']:.0f}g CO2."
            )
            
        if kpis.get('eco_score', 100) < 70:
            recommendations.append(
                f"🌱 ÉCO-SCORE: {kpis['eco_score']}/100. "
                f"Conseils: vitesse stable, anticipation, accélérations douces."
            )
        
        if kpis.get('events_per_km', 0) > 3:
            recommendations.append(
                f"⚠️ CONDUITE: Taux d'événements élevé ({kpis['events_per_km']:.1f}/km). "
                f"Conduite plus fluide = sécurité + économies."
            )
        
        return recommendations
    
    def close(self) -> Dict:
        """Ferme le logger et sauvegarde le résumé"""
        self.csv_file.close()
        summary = self.save_summary()
        return summary


if __name__ == "__main__":
    # Test
    logger = SessionLogger(session_id='test_001')
    
    # Simuler quelques datapoints
    for i in range(100):
        logger.log_datapoint(
            fatigue_metrics={'composite_score': i * 0.5, 'eye_ratio': 0.25, 'yawn_count': i // 20},
            driving_data={'speed_kmh': 80 + i % 20, 'acceleration_ms2': 0.5, 'co2_estimate_gkm': 120},
            analysis={'events': ['harsh_acceleration'] if i % 30 == 0 else [], 'total_distance_km': i * 0.02, 'eco_score': 85}
        )
    
    summary = logger.close()
    print(f"✅ Session saved: {logger.csv_path}")
    print(f"   Summary: {logger.summary_path}")
