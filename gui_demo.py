"""
GUI Demo - Interface graphique pour le système de monitoring conducteur
Interface simple et professionnelle avec Tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
import sys
import os

# Import des modules du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Fatigue detection'))

from simulation import DrivingSimulator
from behavior_analyzer import BehaviorAnalyzer
from session_logger import SessionLogger


class DriverMonitoringGUI:
    """Interface graphique pour le système de monitoring"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Yaqza AI - Smart Driver Monitoring System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.is_running = False
        self.session_thread = None
        self.simulator = None
        self.analyzer = None
        self.logger = None
        self.start_time = None
        
        # Configuration style
        self.setup_styles()
        
        # Créer l'interface
        self.create_header()
        self.create_metrics_panel()
        self.create_status_panel()
        self.create_behavior_panel()
        self.create_events_panel()
        self.create_control_panel()
        self.create_footer()
        
    def setup_styles(self):
        """Configure les styles de l'interface"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Couleurs
        self.colors = {
            'primary': '#2196F3',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#F44336',
            'dark': '#212121',
            'light': '#FFFFFF'
        }
        
    def create_header(self):
        """Crée l'en-tête"""
        header = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)
        
        title = tk.Label(
            header,
            text="Yaqza AI - Alert Before Danger",
            font=('Segoe UI', 24, 'bold'),
            bg=self.colors['primary'],
            fg=self.colors['light']
        )
        title.pack(pady=20)
        
        subtitle = tk.Label(
            header,
            text="Smart Driver Monitoring & Eco-Driving System",
            font=('Segoe UI', 11),
            bg=self.colors['primary'],
            fg=self.colors['light']
        )
        subtitle.pack()
        
    def create_metrics_panel(self):
        """Crée le panneau des métriques principales"""
        metrics_frame = tk.Frame(self.root, bg='#f0f0f0')
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Fatigue Score
        self.fatigue_frame = self.create_metric_card(
            metrics_frame,
            "Fatigue Score",
            "0",
            "/100",
            self.colors['success']
        )
        self.fatigue_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        
        # Eco Score
        self.eco_frame = self.create_metric_card(
            metrics_frame,
            "Eco Score",
            "100",
            "/100",
            self.colors['success']
        )
        self.eco_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        
        # Speed
        self.speed_frame = self.create_metric_card(
            metrics_frame,
            "Vitesse",
            "0",
            "km/h",
            self.colors['primary']
        )
        self.speed_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        
        # CO2
        self.co2_frame = self.create_metric_card(
            metrics_frame,
            "CO2",
            "0",
            "g/km",
            self.colors['warning']
        )
        self.co2_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        
    def create_metric_card(self, parent, title, value, unit, color):
        """Crée une carte de métrique"""
        frame = tk.Frame(parent, bg=self.colors['light'], relief=tk.RAISED, borderwidth=1)
        
        title_label = tk.Label(
            frame,
            text=title,
            font=('Segoe UI', 10),
            bg=self.colors['light'],
            fg=self.colors['dark']
        )
        title_label.pack(pady=(10, 5))
        
        value_frame = tk.Frame(frame, bg=self.colors['light'])
        value_frame.pack()
        
        value_label = tk.Label(
            value_frame,
            text=value,
            font=('Segoe UI', 32, 'bold'),
            bg=self.colors['light'],
            fg=color
        )
        value_label.pack(side=tk.LEFT)
        
        unit_label = tk.Label(
            value_frame,
            text=unit,
            font=('Segoe UI', 12),
            bg=self.colors['light'],
            fg=self.colors['dark']
        )
        unit_label.pack(side=tk.LEFT, padx=(5, 0), pady=(10, 0))
        
        # Stocker les labels pour mise à jour
        frame.value_label = value_label
        frame.title_label = title_label
        
        return frame
        
    def create_status_panel(self):
        """Crée le panneau de statut"""
        status_frame = tk.LabelFrame(
            self.root,
            text="Statut du Système",
            font=('Segoe UI', 11, 'bold'),
            bg='#f0f0f0',
            fg=self.colors['dark']
        )
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        inner_frame = tk.Frame(status_frame, bg=self.colors['light'])
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Session info
        self.session_label = tk.Label(
            inner_frame,
            text="Session: Pas de session active",
            font=('Segoe UI', 10),
            bg=self.colors['light'],
            fg=self.colors['dark'],
            anchor='w'
        )
        self.session_label.pack(fill=tk.X, pady=2)
        
        self.duration_label = tk.Label(
            inner_frame,
            text="Durée: 00:00:00",
            font=('Segoe UI', 10),
            bg=self.colors['light'],
            fg=self.colors['dark'],
            anchor='w'
        )
        self.duration_label.pack(fill=tk.X, pady=2)
        
        self.distance_label = tk.Label(
            inner_frame,
            text="Distance: 0.00 km",
            font=('Segoe UI', 10),
            bg=self.colors['light'],
            fg=self.colors['dark'],
            anchor='w'
        )
        self.distance_label.pack(fill=tk.X, pady=2)
        
        self.events_label = tk.Label(
            inner_frame,
            text="Événements: 0",
            font=('Segoe UI', 10),
            bg=self.colors['light'],
            fg=self.colors['dark'],
            anchor='w'
        )
        self.events_label.pack(fill=tk.X, pady=2)
        
    def create_behavior_panel(self):
        """Crée le panneau d'analyse de conduite (éco + événements)"""
        behavior_frame = tk.LabelFrame(
            self.root,
            text="Analyse de Conduite (Éco)",
            font=('Segoe UI', 11, 'bold'),
            bg='#f0f0f0',
            fg=self.colors['dark']
        )
        behavior_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        inner = tk.Frame(behavior_frame, bg=self.colors['light'])
        inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Eco-score courant
        self.eco_score_label = tk.Label(
            inner,
            text="Éco-score: 100/100",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['light'],
            fg=self.colors['dark'],
            anchor='w'
        )
        self.eco_score_label.grid(row=0, column=0, sticky='w', pady=(0, 6))

        # Compteurs d'événements par type
        self.event_labels = {}
        labels = [
            ("harsh_acceleration", "Accélérations brusques"),
            ("harsh_braking", "Freinages brusques"),
            ("speeding", "Excès de vitesse"),
            ("idling", "Ralenti prolongé"),
            ("steering_oscillation", "Oscillations volant"),
        ]
        for i, (key, title) in enumerate(labels, start=1):
            lbl = tk.Label(
                inner,
                text=f"{title}: 0",
                font=('Segoe UI', 9),
                bg=self.colors['light'],
                fg=self.colors['dark'],
                anchor='w'
            )
            lbl.grid(row=i, column=0, sticky='w', pady=2)
            self.event_labels[key] = lbl

        # Mise en forme grille
        inner.columnconfigure(0, weight=1)

    def create_events_panel(self):
        """Crée le panneau des événements et recommandations"""
        events_frame = tk.LabelFrame(
            self.root,
            text="Événements & Recommandations",
            font=('Segoe UI', 11, 'bold'),
            bg='#f0f0f0',
            fg=self.colors['dark']
        )
        events_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(events_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Liste des événements
        self.events_listbox = tk.Listbox(
            events_frame,
            font=('Segoe UI', 9),
            bg=self.colors['light'],
            fg=self.colors['dark'],
            yscrollcommand=scrollbar.set,
            height=8
        )
        self.events_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.events_listbox.yview)
        
    def create_control_panel(self):
        """Crée le panneau de contrôle"""
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Mode de conduite
        mode_frame = tk.Frame(control_frame, bg='#f0f0f0')
        mode_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            mode_frame,
            text="Mode:",
            font=('Segoe UI', 10),
            bg='#f0f0f0'
        ).pack(side=tk.LEFT, padx=5)
        
        self.mode_var = tk.StringVar(value='mixed')
        mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=['highway', 'urban', 'mixed'],
            state='readonly',
            width=10,
            font=('Segoe UI', 10)
        )
        mode_combo.pack(side=tk.LEFT)
        
        # Boutons
        button_frame = tk.Frame(control_frame, bg='#f0f0f0')
        button_frame.pack(side=tk.RIGHT, padx=10)
        
        self.start_button = tk.Button(
            button_frame,
            text="Démarrer Session",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['success'],
            fg=self.colors['light'],
            activebackground='#45a049',
            command=self.start_session,
            width=15,
            height=2,
            cursor='hand2'
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(
            button_frame,
            text="Arrêter Session",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['danger'],
            fg=self.colors['light'],
            activebackground='#da190b',
            command=self.stop_session,
            width=15,
            height=2,
            state=tk.DISABLED,
            cursor='hand2'
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
    def create_footer(self):
        """Crée le pied de page"""
        footer = tk.Frame(self.root, bg=self.colors['dark'], height=40)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        
        footer_text = tk.Label(
            footer,
            text="Hack For Good 4.0 | Yaqza AI © 2025 | Alert Before Danger",
            font=('Segoe UI', 9),
            bg=self.colors['dark'],
            fg=self.colors['light']
        )
        footer_text.pack(pady=10)
        
    def start_session(self):
        """Démarre une session de monitoring"""
        if self.is_running:
            return
            
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Initialiser les composants
        mode = self.mode_var.get()
        self.simulator = DrivingSimulator(mode=mode)
        self.analyzer = BehaviorAnalyzer()
        self.logger = SessionLogger()
        self.start_time = time.time()
        
        # Mettre à jour l'interface
        self.session_label.config(text=f"Session: {self.logger.session_id}")
        self.add_event("INFO", "Session démarrée en mode " + mode)
        
        # Démarrer le thread de monitoring
        self.session_thread = threading.Thread(target=self.run_monitoring, daemon=True)
        self.session_thread.start()
        
    def stop_session(self):
        """Arrête la session en cours"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        # Sauvegarder la session
        if self.logger:
            summary = self.logger.close()
            self.add_event("INFO", f"Session sauvegardée: {self.logger.csv_path}")
            
            # Afficher résumé
            kpis = summary['kpis']
            message = f"""Session terminée!

Durée: {kpis['duration_minutes']:.1f} min
Distance: {kpis['distance_km']} km
Eco-score final: {kpis['eco_score']}/100
Événements: {kpis['total_events']}

Fichiers sauvegardés:
- {self.logger.csv_path}
- {self.logger.summary_path}
"""
            messagebox.showinfo("Session Terminée", message)
        
    def run_monitoring(self):
        """Boucle principale de monitoring (thread séparé)"""
        iteration = 0
        
        while self.is_running:
            iteration += 1
            elapsed = time.time() - self.start_time
            
            # Simuler fatigue progressive
            time_factor = elapsed / 300  # 5 minutes
            simulated_fatigue = min(100, time_factor ** 1.5 * 120)
            
            fatigue_metrics = {
                'composite_score': simulated_fatigue,
                'eye_ratio': 0.3,
                'yawn_count': int(simulated_fatigue / 20),
                'head_pitch': 0,
                'head_yaw': 0,
                'head_roll': 0
            }
            
            # Générer données de conduite
            driving_data = self.simulator.update(fatigue_score=simulated_fatigue)
            
            # Analyser
            events, metrics = self.analyzer.analyze_datapoint(driving_data)
            eco_score = self.analyzer.calculate_eco_score()
            
            # Logger
            analysis = {
                'events': events,
                'total_distance_km': metrics['total_distance_km'],
                'eco_score': eco_score,
                'recommendation': ''
            }
            
            self.logger.log_datapoint(fatigue_metrics, driving_data, analysis)
            
            # Mettre à jour l'interface (dans le thread principal)
            self.root.after(0, self.update_interface, 
                          simulated_fatigue, 
                          eco_score, 
                          driving_data, 
                          events, 
                          metrics, 
                          elapsed)
            
            time.sleep(1)  # 1 Hz
            
    def update_interface(self, fatigue, eco_score, driving, events, metrics, elapsed):
        """Met à jour l'interface avec les nouvelles données"""
        # Métriques
        fatigue_color = self.get_fatigue_color(fatigue)
        self.fatigue_frame.value_label.config(text=f"{int(fatigue)}", fg=fatigue_color)
        
        eco_color = self.get_eco_color(eco_score)
        self.eco_frame.value_label.config(text=f"{int(eco_score)}", fg=eco_color)
        
        self.speed_frame.value_label.config(text=f"{int(driving['speed_kmh'])}")
        self.co2_frame.value_label.config(text=f"{int(driving['co2_estimate_gkm'])}")
        
        # Statut
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        self.duration_label.config(text=f"Durée: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        self.distance_label.config(text=f"Distance: {metrics['total_distance_km']:.2f} km")
        
        total_events = sum(self.analyzer.event_counts.values())
        self.events_label.config(text=f"Événements: {total_events}")

        # Mise à jour panneau éco + événements
        self.eco_score_label.config(text=f"Éco-score: {int(eco_score)}/100")
        for key, lbl in self.event_labels.items():
            count = self.analyzer.event_counts.get(key, 0)
            lbl.config(text=f"{lbl.cget('text').split(':')[0]}: {count}")
        
        # Événements
        if events:
            for event in events:
                if event in ['harsh_acceleration', 'harsh_braking', 'speeding']:
                    self.add_event("ALERTE", self.format_event(event, driving))
                else:
                    self.add_event("INFO", self.format_event(event, driving))
        
        # Recommandation fatigue
        if fatigue > 70 and int(elapsed) % 10 == 0:
            self.add_event("CRITIQUE", "FATIGUE CRITIQUE - Pause immédiate recommandée!")
        elif fatigue > 40 and int(elapsed) % 30 == 0:
            self.add_event("ATTENTION", "Signes de fatigue détectés - Planifiez une pause")
            
    def get_fatigue_color(self, score):
        """Retourne la couleur selon le score de fatigue"""
        if score < 40:
            return self.colors['success']
        elif score < 70:
            return self.colors['warning']
        else:
            return self.colors['danger']
            
    def get_eco_color(self, score):
        """Retourne la couleur selon l'éco-score"""
        if score >= 80:
            return self.colors['success']
        elif score >= 60:
            return self.colors['warning']
        else:
            return self.colors['danger']
            
    def format_event(self, event, driving):
        """Formate un événement pour affichage"""
        messages = {
            'harsh_acceleration': f"Accélération brusque détectée ({driving['acceleration_ms2']:.1f} m/s²)",
            'harsh_braking': f"Freinage brusque détecté ({driving['acceleration_ms2']:.1f} m/s²)",
            'speeding': f"Excès de vitesse ({driving['speed_kmh']:.0f} km/h)",
            'idling': "Ralenti prolongé détecté",
            'steering_oscillation': f"Oscillations volant ({driving['steering_angle_deg']:.1f}°)"
        }
        return messages.get(event, event)
        
    def add_event(self, level, message):
        """Ajoute un événement à la liste"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        event_text = f"[{timestamp}] [{level}] {message}"
        
        self.events_listbox.insert(0, event_text)
        
        # Limiter à 100 événements
        if self.events_listbox.size() > 100:
            self.events_listbox.delete(tk.END)
            
        # Couleur selon niveau
        if level == "CRITIQUE":
            self.events_listbox.itemconfig(0, fg=self.colors['danger'])
        elif level == "ALERTE":
            self.events_listbox.itemconfig(0, fg=self.colors['warning'])


def main():
    """Point d'entrée principal"""
    root = tk.Tk()
    app = DriverMonitoringGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
