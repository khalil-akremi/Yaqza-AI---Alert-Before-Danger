"""
Integrated Demo - Démo complète du système de monitoring conducteur + éco-conduite
Combine détection fatigue (webcam) + simulation télémétrie + analyse temps réel
"""

import cv2
import time
import threading
from datetime import datetime
import sys
import os
from queue import Queue
try:
    import pyttsx3  # TTS pour alertes vocales
except ImportError:
    pyttsx3 = None

# Import des modules du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Fatigue detection'))

from simulation import DrivingSimulator
from behavior_analyzer import BehaviorAnalyzer
from session_logger import SessionLogger


class IntegratedDriverMonitoring:
    """Système intégré de monitoring conducteur"""
    
    def __init__(self, mode='mixed', use_camera=True):
        """
        Args:
            mode: Mode de simulation ('highway', 'urban', 'mixed')
            use_camera: Utiliser la caméra pour détection fatigue
        """
        self.mode = mode
        self.use_camera = use_camera
        
        # Composants
        self.simulator = DrivingSimulator(mode=mode)
        self.analyzer = BehaviorAnalyzer()
        self.logger = SessionLogger()
        
        # Historique pour KPIs phase 1
        self.fatigue_history = []
        self.eco_history = []
        self.event_history = []
        
        # État fatigue (simulé si pas de caméra)
        self.fatigue_metrics = {
            'composite_score': 0,
            'eye_ratio': 0.3,
            'yawn_count': 0,
            'head_pitch': 0,
            'head_yaw': 0,
            'head_roll': 0
        }
        
        # Thread de capture caméra
        self.camera_thread = None
        self.running = False
        
        if self.use_camera:
            try:
                # Essayer d'importer le détecteur de fatigue
                from sleepy_detector import FatigueDetector
                self.fatigue_detector = FatigueDetector()
                print("✅ Détection fatigue activée (caméra)")
            except Exception as e:
                print(f"⚠️ Caméra non disponible: {e}")
                print("   Mode simulation fatigue activé")
                self.use_camera = False
        
        # Initialisation alertes vocales (Phase 1)
        self.voice_enabled = False
        self.voice_queue: Queue[str] = Queue()
        self.voice_cooldowns = {
            'critical_fatigue': 30,
            'high_fatigue': 60,
            'harsh_events': 45,
            'eco_drop': 120
        }
        self.last_voice_time = {k: 0 for k in self.voice_cooldowns.keys()}
        self.voice_thread = None
        if pyttsx3:
            try:
                self.voice_engine = pyttsx3.init()
                self.voice_engine.setProperty('rate', 175)
                self.voice_enabled = True
                print("🔊 Alertes vocales activées")
            except Exception as e:
                print(f"⚠️ TTS indisponible: {e}")
                self.voice_enabled = False
        
    def process_fatigue(self):
        """Thread séparé pour traiter la détection de fatigue"""
        while self.running:
            # Ici on mettrait la logique de capture webcam
            # Pour l'instant, simulation
            time.sleep(0.1)
    
    def run(self, duration_seconds=120):
        """
        Lance la démo intégrée
        
        Args:
            duration_seconds: Durée de la démo en secondes
        """
        self.running = True
        
        # Démarrer thread caméra si activé
        if self.use_camera:
            self.camera_thread = threading.Thread(target=self.process_fatigue)
            self.camera_thread.start()
        
        # Thread vocal
        if self.voice_enabled:
            self.voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
            self.voice_thread.start()
        
        print("=" * 70)
        print("🚗 SYSTÈME INTÉGRÉ DE SURVEILLANCE CONDUCTEUR & ÉCO-CONDUITE")
        print("=" * 70)
        print(f"Session ID: {self.logger.session_id}")
        print(f"Mode conduite: {self.mode}")
        print(f"Durée: {duration_seconds}s (~{duration_seconds/60:.1f} min)")
        print()
        print("Appuyez sur ESC pour arrêter")
        print("=" * 70)
        print()
        
        start_time = time.time()
        iteration = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                iteration += 1
                
                # 1. Simuler augmentation fatigue progressive
                elapsed = time.time() - start_time
                time_factor = elapsed / duration_seconds
                simulated_fatigue = min(100, time_factor ** 1.5 * 120)
                self.fatigue_metrics['composite_score'] = simulated_fatigue
                
                # 2. Générer données de conduite (influencées par fatigue)
                driving_data = self.simulator.update(fatigue_score=simulated_fatigue)
                
                # 3. Analyser comportement
                events, metrics = self.analyzer.analyze_datapoint(driving_data)
                
                # 4. Calculer recommandation
                recommendation = self._generate_instant_recommendation(
                    self.fatigue_metrics, 
                    driving_data, 
                    events
                )
                
                # 5. Logger
                analysis = {
                    'events': events,
                    'total_distance_km': metrics['total_distance_km'],
                    'eco_score': self.analyzer.calculate_eco_score(),
                    'recommendation': recommendation
                }
                
                self.logger.log_datapoint(
                    self.fatigue_metrics,
                    driving_data,
                    analysis
                )
                # Historique pour KPIs rapides
                self.fatigue_history.append(self.fatigue_metrics['composite_score'])
                self.eco_history.append(analysis['eco_score'])
                self.event_history.append(len(events))
                
                # Alertes vocales Phase 1
                self._evaluate_voice_alerts(self.fatigue_metrics['composite_score'], analysis['eco_score'], events)
                
                # 6. Affichage temps réel (toutes les secondes)
                if iteration % 1 == 0:
                    self._display_status(
                        int(elapsed),
                        self.fatigue_metrics,
                        driving_data,
                        events,
                        self.analyzer.calculate_eco_score(),
                        recommendation
                    )
                
                time.sleep(1)  # 1 Hz
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Arrêt demandé par l'utilisateur")
        
        finally:
            self.running = False
            if self.camera_thread:
                self.camera_thread.join()
            if self.voice_thread and self.voice_enabled:
                # Vider file et terminer
                self.voice_enabled = False
                self.voice_queue.put(None)
            
            # Sauvegarder et afficher résumé
            print("\n" + "=" * 70)
            print("💾 Saving session data...")
            summary = self.logger.close()
            
            print(f"\n✅ Session {self.logger.session_id} saved:")
            print(f"   Data: {self.logger.csv_path}")
            print(f"   Summary: {self.logger.summary_path}")
            
            self._display_summary(summary)
    
    def _generate_instant_recommendation(self, fatigue, driving, events):
        """Génère une recommandation instantanée"""
        if fatigue['composite_score'] > 70:
            return "PAUSE IMMÉDIATE - Fatigue critique détectée!"
        
        if 'harsh_acceleration' in events:
            return "Accélérations plus progressives pour réduire consommation"
        
        if 'harsh_braking' in events:
            return "Anticiper le trafic pour freiner en douceur"
        
        if 'steering_oscillation' in events:
            return "Maintenir trajectoire stable - signe possible de fatigue?"
        
        if driving['co2_estimate_gkm'] > 130:
            return f"CO2 élevé ({driving['co2_estimate_gkm']:.0f}g/km) - réduire vitesse/accélérations"
        
        return ""

    # ====== Alertes Vocales (Phase 1) ======
    def _voice_loop(self):
        """Consomme la file d'attente des messages vocaux"""
        while self.voice_enabled:
            msg = self.voice_queue.get()
            if msg is None:
                break
            try:
                self.voice_engine.say(msg)
                self.voice_engine.runAndWait()
            except Exception:
                pass
            time.sleep(0.2)

    def _queue_voice(self, key: str, message: str):
        """Ajoute un message vocal selon cooldown"""
        now = time.time()
        cooldown = self.voice_cooldowns.get(key, 60)
        if now - self.last_voice_time[key] >= cooldown:
            self.last_voice_time[key] = now
            self.voice_queue.put(message)

    def _evaluate_voice_alerts(self, fatigue_score: float, eco_score: float, events: list):
        """Détermine quelles alertes vocales déclencher"""
        if not self.voice_enabled:
            return
        # Fatigue critique
        if fatigue_score >= 70:
            self._queue_voice('critical_fatigue', 'Fatigue critique détectée. Veuillez vous arrêter immédiatement et prendre une pause.')
        elif fatigue_score >= 40:
            self._queue_voice('high_fatigue', 'Niveau de fatigue élevé. Préparez une pause prochainement.')
        # Événements brusques répétés
        harsh = sum(1 for e in events if e in ['harsh_acceleration', 'harsh_braking'])
        if harsh > 0:
            self._queue_voice('harsh_events', 'Conduite brusque détectée. Accélérez et freinez plus progressivement.')
        # Baisse éco-score
        if eco_score < 60:
            self._queue_voice('eco_drop', 'Score éco faible. Adoptez une vitesse stable et évitez les accélérations fortes.')
    
    def _display_status(self, elapsed_sec, fatigue, driving, events, eco_score, recommendation):
        """Affiche le statut en temps réel"""
        fatigue_score = fatigue['composite_score']
        
        # Indicateur fatigue
        if fatigue_score < 40:
            fatigue_icon = "🟢"
            fatigue_status = "OK"
        elif fatigue_score < 70:
            fatigue_icon = "🟠"
            fatigue_status = "ATTENTION"
        else:
            fatigue_icon = "🔴"
            fatigue_status = "CRITIQUE"
        
        # Indicateur éco
        if eco_score >= 80:
            eco_icon = "🌱"
        elif eco_score >= 60:
            eco_icon = "🟡"
        else:
            eco_icon = "🔴"
        
        print(f"⏱️  [{elapsed_sec}s] {fatigue_icon} Fatigue: {fatigue_status} ({fatigue_score:.0f}/100) | {eco_icon} Éco: {eco_score:.0f}/100")
        print(f"   Vitesse: {driving['speed_kmh']:.0f} km/h | CO2: {driving['co2_estimate_gkm']:.0f} g/km | Bâillements: {fatigue['yawn_count']}")
        
        # Événements
        if events:
            event_msg = ", ".join(events)
            if 'harsh' in event_msg or 'speeding' in event_msg:
                print(f"   ⚠️ {event_msg}")
            else:
                print(f"   ℹ️ {event_msg}")
        
        # Recommandation
        if recommendation:
            print(f"      💡 {recommendation}")
        
        print()
    
    def _display_summary(self, summary):
        """Affiche le résumé final"""
        kpis = summary['kpis']
        
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ DE SESSION")
        print("=" * 70)
        print()
        print(f"⏱️  Durée: {kpis['duration_minutes']} minutes")
        print(f"🛣️  Distance: {kpis['distance_km']} km")
        print(f"🚗 Vitesse moyenne: {kpis['avg_speed_kmh']} km/h")
        
        print(f"\n😴 FATIGUE:")
        print(f"   • Score max: {kpis['max_fatigue_score']}/100")
        print(f"   • Score moyen: {kpis['avg_fatigue_score']}/100")
        print(f"   • Temps avant fatigue: {kpis['time_to_fatigue_min']} min")
        print(f"   • Bâillements totaux: {kpis['total_yawns']}")
        
        print(f"\n🌱 ÉCO-CONDUITE:")
        print(f"   • Score éco: {kpis['eco_score']}/100")
        print(f"   • CO2 moyen: {kpis['avg_co2_gkm']} g/km (optimal: 110 g/km)")
        print(f"   • Excès CO2: {kpis['co2_excess_pct']}%")
        print(f"   • CO2 économisable: {kpis['potential_co2_savings_g']:.0f}g")
        
        print(f"\n⚠️  ÉVÉNEMENTS:")
        print(f"   • Total: {kpis['total_events']} ({kpis['events_per_km']}/km)")
        for event_type, count in kpis['event_breakdown'].items():
            print(f"   • {event_type}: {count}")
        
        print(f"\n💡 RECOMMANDATIONS:")
        for rec in summary['recommendations']:
            print(f"   {rec}")
        
        # Ajout KPIs Phase 1 rapides
        if self.fatigue_history:
            slope = (self.fatigue_history[-1] - self.fatigue_history[0]) / max(1, len(self.fatigue_history))
            avg_eco = sum(self.eco_history)/len(self.eco_history) if self.eco_history else 0
            print("\n📈 KPIs supplémentaires:")
            print(f"   • Slope fatigue (approx): {slope:.2f} / tick")
            print(f"   • Eco-score moyen session: {avg_eco:.1f}/100")
            if avg_eco < 70:
                print("   • Suggestion: Réduire variations brusques pour remonter l'éco-score")
        
        if self.voice_enabled:
            print("\n🔊 Alertes vocales actives durant la session")
        
        print("\n" + "=" * 70)
        print("📁 Fichiers sauvegardés:")
        print(f"   • {self.logger.csv_path}")
        print(f"   • {self.logger.summary_path}")
        print("=" * 70)


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Démo intégrée système monitoring conducteur')
    parser.add_argument('--duration', type=int, default=120, help='Durée en secondes (défaut: 120)')
    parser.add_argument('--mode', choices=['highway', 'urban', 'mixed'], default='mixed', 
                        help='Mode de conduite (défaut: mixed)')
    parser.add_argument('--no-camera', action='store_true', help='Désactiver la caméra')
    
    args = parser.parse_args()
    
    system = IntegratedDriverMonitoring(mode=args.mode, use_camera=not args.no_camera)
    system.run(duration_seconds=args.duration)


if __name__ == "__main__":
    main()
