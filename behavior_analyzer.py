"""
Behavior Analyzer - Détection d'événements et scoring éco-conduite
Analyse télémétrie pour détecter comportements à risque et calculer éco-score
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class BehaviorAnalyzer:
    """Analyseur de comportement de conduite en temps réel"""
    
    def __init__(self):
        # Seuils configurables
        self.harsh_accel_threshold = 2.5  # m/s²
        self.harsh_brake_threshold = -2.5  # m/s²
        self.speed_limit_urban = 50  # km/h
        self.speed_limit_highway = 130  # km/h
        self.idling_speed_threshold = 2  # km/h
        self.idling_throttle_threshold = 10  # %
        self.steering_oscillation_threshold = 15  # deg variation
        
        # Historique pour détections temporelles
        self.history = []
        self.events = []
        
        # Compteurs de session
        self.total_distance_km = 0
        self.total_co2_g = 0
        self.event_counts = {
            'harsh_acceleration': 0,
            'harsh_braking': 0,
            'speeding': 0,
            'idling': 0,
            'steering_oscillation': 0
        }
        
    def analyze_datapoint(self, data: Dict) -> Tuple[List[str], Dict]:
        """
        Analyse un point de données en temps réel
        
        Args:
            data: Dict avec clés speed_kmh, acceleration_ms2, etc.
            
        Returns:
            (events, metrics) où events est une liste de strings et metrics un dict
        """
        events_detected = []
        
        # Extraction des valeurs
        speed = data.get('speed_kmh', 0)
        accel = data.get('acceleration_ms2', 0)
        steering = data.get('steering_angle_deg', 0)
        throttle = data.get('throttle_position_pct', 0)
        co2 = data.get('co2_estimate_gkm', 0)
        
        # 1. Accélération brusque
        if accel > self.harsh_accel_threshold:
            events_detected.append('harsh_acceleration')
            self.event_counts['harsh_acceleration'] += 1
            self.events.append({
                'type': 'harsh_acceleration',
                'value': accel,
                'timestamp': data.get('timestamp', '')
            })
            
        # 2. Freinage brusque
        if accel < self.harsh_brake_threshold:
            events_detected.append('harsh_braking')
            self.event_counts['harsh_braking'] += 1
            self.events.append({
                'type': 'harsh_braking',
                'value': accel,
                'timestamp': data.get('timestamp', '')
            })
            
        # 3. Excès de vitesse (contexte-aware)
        speed_limit = self.speed_limit_highway if speed > 80 else self.speed_limit_urban
        if speed > speed_limit * 1.1:  # +10% tolérance
            events_detected.append('speeding')
            self.event_counts['speeding'] += 1
            
        # 4. Ralenti prolongé (idling)
        if speed < self.idling_speed_threshold and throttle > self.idling_throttle_threshold:
            events_detected.append('idling')
            self.event_counts['idling'] += 1
            
        # 5. Oscillations volant (fatigue indicator)
        if len(self.history) > 0:
            prev_steering = self.history[-1].get('steering_angle_deg', 0)
            steering_change = abs(steering - prev_steering)
            if steering_change > self.steering_oscillation_threshold:
                events_detected.append('steering_oscillation')
                self.event_counts['steering_oscillation'] += 1
        
        # Mise à jour statistiques
        if speed > 0:
            distance_m = speed / 3.6  # m/s
            self.total_distance_km += distance_m / 1000
            self.total_co2_g += (co2 * distance_m) / 1000
        
        # Historique (garder 10 derniers points)
        self.history.append(data)
        if len(self.history) > 10:
            self.history.pop(0)
            
        # Métriques
        metrics = {
            'events': events_detected,
            'event_counts': self.event_counts.copy(),
            'total_distance_km': round(self.total_distance_km, 2),
            'avg_co2_gkm': round(self.total_co2_g / max(self.total_distance_km, 0.001), 1)
        }
        
        return events_detected, metrics
    
    def calculate_eco_score(self) -> float:
        """
        Calcule un score éco-conduite 0-100
        Plus le score est élevé, meilleure est la conduite
        """
        if self.total_distance_km < 0.1:
            return 100.0
            
        # Pénalités par type d'événement (sur 100 points)
        penalties = 0
        
        # Pénalité par événement / km
        events_per_km = sum(self.event_counts.values()) / self.total_distance_km
        penalties += min(events_per_km * 5, 40)  # Max -40 pts
        
        # Pénalité CO2 excessif
        optimal_co2 = 110  # g/km optimal
        avg_co2 = self.total_co2_g / self.total_distance_km
        if avg_co2 > optimal_co2:
            co2_excess_pct = ((avg_co2 - optimal_co2) / optimal_co2) * 100
            penalties += min(co2_excess_pct * 0.5, 30)  # Max -30 pts
        
        # Pénalités spécifiques
        harsh_events = self.event_counts['harsh_acceleration'] + self.event_counts['harsh_braking']
        penalties += min(harsh_events / self.total_distance_km * 3, 20)  # Max -20 pts
        
        idling_penalty = min(self.event_counts['idling'] / max(self.total_distance_km, 1) * 2, 10)
        penalties += idling_penalty  # Max -10 pts
        
        score = max(0, 100 - penalties)
        return round(score, 1)
    
    def get_session_summary(self) -> Dict:
        """Retourne un résumé complet de la session"""
        eco_score = self.calculate_eco_score()
        avg_co2 = self.total_co2_g / max(self.total_distance_km, 0.001)
        optimal_co2 = 110
        
        co2_excess_pct = ((avg_co2 - optimal_co2) / optimal_co2) * 100
        co2_savings_g = max(0, self.total_co2_g - (optimal_co2 * self.total_distance_km))
        
        return {
            'distance_km': round(self.total_distance_km, 2),
            'eco_score': eco_score,
            'total_co2_g': round(self.total_co2_g, 1),
            'avg_co2_gkm': round(avg_co2, 1),
            'co2_excess_pct': round(co2_excess_pct, 1),
            'potential_co2_savings_g': round(co2_savings_g, 1),
            'events': self.event_counts.copy(),
            'total_events': sum(self.event_counts.values()),
            'events_per_km': round(sum(self.event_counts.values()) / max(self.total_distance_km, 0.001), 2)
        }


def analyze_csv_dataset(input_file='driving_dataset.csv', output_file='driving_dataset_analyzed.csv'):
    """
    Analyse un dataset CSV complet et ajoute colonnes d'analyse
    
    Args:
        input_file: Fichier CSV d'entrée
        output_file: Fichier CSV de sortie avec colonnes supplémentaires
    """
    print(f"📊 Analyse du dataset: {input_file}")
    
    df = pd.read_csv(input_file)
    analyzer = BehaviorAnalyzer()
    
    # Nouvelles colonnes
    df['events_detected'] = ''
    df['event_count'] = 0
    df['cumulative_distance_km'] = 0.0
    df['cumulative_co2_g'] = 0.0
    df['eco_score'] = 100.0
    
    # Analyse ligne par ligne
    for idx, row in df.iterrows():
        data = row.to_dict()
        events, metrics = analyzer.analyze_datapoint(data)
        
        df.at[idx, 'events_detected'] = ','.join(events) if events else 'none'
        df.at[idx, 'event_count'] = len(events)
        df.at[idx, 'cumulative_distance_km'] = metrics['total_distance_km']
        df.at[idx, 'cumulative_co2_g'] = analyzer.total_co2_g
        df.at[idx, 'eco_score'] = analyzer.calculate_eco_score()
    
    # Sauvegarder
    df.to_csv(output_file, index=False)
    
    # Résumé
    summary = analyzer.get_session_summary()
    print(f"\n✅ Analyse terminée: {output_file}")
    print(f"\n📈 RÉSUMÉ DE SESSION:")
    print(f"   Distance totale: {summary['distance_km']} km")
    print(f"   Score éco-conduite: {summary['eco_score']}/100")
    print(f"   CO2 total: {summary['total_co2_g']}g")
    print(f"   CO2 moyen: {summary['avg_co2_gkm']} g/km")
    print(f"   Excès CO2: {summary['co2_excess_pct']}%")
    print(f"   Économie potentielle: {summary['potential_co2_savings_g']}g")
    print(f"\n⚠️  ÉVÉNEMENTS DÉTECTÉS ({summary['total_events']} total, {summary['events_per_km']}/km):")
    for event_type, count in summary['events'].items():
        if count > 0:
            print(f"   • {event_type}: {count}")
    
    return df, summary


if __name__ == "__main__":
    import sys
    
    input_file = 'driving_dataset.csv'
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
    analyze_csv_dataset(input_file)
