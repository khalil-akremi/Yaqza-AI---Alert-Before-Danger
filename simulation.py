"""
Driving Telemetry Simulator - Génération de données de conduite réalistes
Simule vitesse, accélération, volant, etc. avec corrélation fatigue
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DrivingSimulator:
    """Simulateur de télémétrie de conduite avec modes réalistes"""
    
    def __init__(self, mode='mixed'):
        """
        Args:
            mode: 'highway', 'urban', or 'mixed'
        """
        self.mode = mode
        self.timestamp = datetime.now()
        
        # Paramètres selon le mode de conduite
        if mode == 'highway':
            self.base_speed = 110  # km/h
            self.speed_variance = 15
            self.accel_variance = 1.5
            self.steering_variance = 5
        elif mode == 'urban':
            self.base_speed = 45
            self.speed_variance = 20
            self.accel_variance = 3.0
            self.steering_variance = 15
        else:  # mixed
            self.base_speed = 70
            self.speed_variance = 25
            self.accel_variance = 2.5
            self.steering_variance = 10
            
        # État initial
        self.speed = self.base_speed
        self.acceleration = 0
        self.steering_angle = 0
        self.throttle_position = 50
        self.brake_pressure = 0
        self.fatigue_level = 0.0
        
    def update(self, fatigue_score=0):
        """
        Génère un point de données à l'instant t
        
        Args:
            fatigue_score: Score de fatigue 0-100 (affecte les paramètres)
        """
        self.timestamp += timedelta(seconds=1)
        
        # Normaliser fatigue 0-1
        self.fatigue_level = min(fatigue_score / 100.0, 1.0)
        
        # Fatigue dégrade les performances
        reaction_degradation = 1 + (self.fatigue_level * 0.5)  # +50% variance max
        steering_stability = 1 + (self.fatigue_level * 2.0)    # +200% instabilité max
        
        # Vitesse avec tendance vers base_speed
        speed_drift = np.random.normal(0, self.speed_variance * reaction_degradation)
        self.speed = np.clip(
            self.speed * 0.95 + self.base_speed * 0.05 + speed_drift,
            max(0, self.base_speed - 30),
            self.base_speed + 30
        )
        
        # Accélération
        self.acceleration = np.random.normal(0, self.accel_variance * reaction_degradation)
        
        # Volant (plus d'oscillations si fatigué)
        steering_change = np.random.normal(0, self.steering_variance * steering_stability)
        self.steering_angle = np.clip(
            self.steering_angle * 0.7 + steering_change,
            -45, 45
        )
        
        # Throttle/Brake
        if self.acceleration > 0.5:
            self.throttle_position = np.clip(50 + self.acceleration * 10, 0, 100)
            self.brake_pressure = 0
        elif self.acceleration < -0.5:
            self.throttle_position = 0
            self.brake_pressure = np.clip(-self.acceleration * 20, 0, 100)
        else:
            self.throttle_position = np.clip(50 + np.random.normal(0, 10), 30, 70)
            self.brake_pressure = 0
            
        return self.get_datapoint()
    
    def get_datapoint(self):
        """Retourne un dict avec toutes les métriques"""
        
        # Calcul CO2 réaliste (g/km)
        # Formule simplifiée: base + facteur vitesse + facteur accélération + pénalités
        base_co2 = 95  # Voiture moyenne
        speed_factor = (self.speed - 50) * 0.3  # +0.3g par km/h au-dessus de 50
        throttle_factor = (self.throttle_position - 50) * 0.2
        accel_penalty = max(0, self.acceleration) * 15  # Pénalité forte pour accélérations
        idling_penalty = 50 if self.speed < 2 and self.throttle_position > 10 else 0
        
        co2_gkm = max(0, base_co2 + speed_factor + throttle_factor + accel_penalty + idling_penalty)
        
        return {
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'speed_kmh': round(self.speed, 1),
            'acceleration_ms2': round(self.acceleration, 2),
            'steering_angle_deg': round(self.steering_angle, 1),
            'throttle_position_pct': round(self.throttle_position, 1),
            'brake_pressure_pct': round(self.brake_pressure, 1),
            'fatigue_score': round(self.fatigue_level * 100, 1),
            'co2_estimate_gkm': round(co2_gkm, 1)
        }


def generate_dataset(duration_minutes=60, mode='mixed', output_file='driving_dataset.csv'):
    """
    Génère un dataset complet de conduite
    
    Args:
        duration_minutes: Durée de la session simulée
        mode: 'highway', 'urban', 'mixed'
        output_file: Nom du fichier CSV de sortie
    """
    simulator = DrivingSimulator(mode=mode)
    data = []
    
    num_points = duration_minutes * 60  # 1 point par seconde
    
    print(f"🚗 Génération de {num_points} points de données ({duration_minutes} min, mode: {mode})...")
    
    for i in range(num_points):
        # Fatigue monte progressivement avec le temps
        # Formule réaliste: lente au début, accélérée après 30min
        time_factor = i / num_points
        fatigue_score = min(100, time_factor ** 1.5 * 120)  # Monte jusqu'à 100
        
        # Variations aléatoires de fatigue
        fatigue_score += np.random.normal(0, 5)
        fatigue_score = np.clip(fatigue_score, 0, 100)
        
        datapoint = simulator.update(fatigue_score=fatigue_score)
        data.append(datapoint)
    
    # Créer DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    # Statistiques
    print(f"\n✅ Dataset généré: {output_file}")
    print(f"   Points: {len(df)}")
    print(f"   Durée: {duration_minutes} min ({duration_minutes/60:.1f}h)")
    print(f"   Vitesse moy: {df['speed_kmh'].mean():.1f} km/h")
    print(f"   Fatigue max: {df['fatigue_score'].max():.1f}/100")
    print(f"   CO2 moyen: {df['co2_estimate_gkm'].mean():.1f} g/km")
    
    # Événements détectés
    harsh_accel = (df['acceleration_ms2'] > 2.5).sum()
    harsh_brake = (df['acceleration_ms2'] < -2.5).sum()
    print(f"   Accélérations brusques: {harsh_accel}")
    print(f"   Freinages brusques: {harsh_brake}")
    
    return df


if __name__ == "__main__":
    import sys
    
    # Paramètres par défaut
    duration = 90  # 90 minutes
    mode = 'mixed'
    
    # Arguments optionnels
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    
    df = generate_dataset(duration_minutes=duration, mode=mode)
