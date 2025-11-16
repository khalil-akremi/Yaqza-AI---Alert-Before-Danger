"""
Yaqza AI - Interface graphique complète avec caméra, détection fatigue et analyse comportementale
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
import sys
import os
import cv2
import dlib
import numpy as np
from PIL import Image, ImageTk

# Import des modules du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Fatigue detection'))

from simulation import DrivingSimulator
from behavior_analyzer import BehaviorAnalyzer
from session_logger import SessionLogger


def landmarks_to_np(landmarks, dtype="int"):
    num = landmarks.num_parts
    coords = np.zeros((num, 2), dtype=dtype)
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords


def _eye_ratio(landmarks: np.ndarray) -> float:
    d1 = np.linalg.norm(landmarks[37] - landmarks[41])
    d2 = np.linalg.norm(landmarks[38] - landmarks[40])
    d3 = np.linalg.norm(landmarks[43] - landmarks[47])
    d4 = np.linalg.norm(landmarks[44] - landmarks[46])
    d_mean = (d1 + d2 + d3 + d4) / 4.0
    d5 = np.linalg.norm(landmarks[36] - landmarks[39])
    d6 = np.linalg.norm(landmarks[42] - landmarks[45])
    d_reference = (d5 + d6) / 2.0
    return d_mean / d_reference if d_reference else 0.0


def _mouth_ratio(landmarks: np.ndarray) -> float:
    v1 = np.linalg.norm(landmarks[51] - landmarks[59])
    v2 = np.linalg.norm(landmarks[53] - landmarks[57])
    v3 = np.linalg.norm(landmarks[50] - landmarks[58])
    vertical_mean = (v1 + v2 + v3) / 3.0
    horizontal = np.linalg.norm(landmarks[48] - landmarks[54])
    return vertical_mean / horizontal if horizontal else 0.0


class YaqzaAIGUI:
    """Interface graphique Yaqza AI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Yaqza AI - Alert Before Danger")
        self.root.geometry("1600x900")
        self.root.configure(bg='#1a1a1a')
        
        # Variables
        self.is_running = False
        self.session_thread = None
        self.camera_thread = None
        self.warning_thread = None
        
        # Components
        self.simulator = None
        self.analyzer = None
        self.logger = None
        self.start_time = None
        
        # Camera & detection
        self.cap = None
        self.detector = None
        self.predictor = None
        self.use_camera = True
        
        # Fatigue tracking
        self.current_fatigue = {
            'composite_score': 0,
            'eye_ratio': 0.3,
            'yawn_count': 0,
            'head_pitch': 0,
            'head_yaw': 0,
            'head_roll': 0
        }
        self.yawn_count = 0
        self.yawn_frame_counter = 0
        self.fatigue_score_smoothed = 0.0
        self.queue = [0] * 60
        self.closed_streak = 0
        
        # Warning system
        self.last_warning_time = 0
        self.warning_cooldown = 5  # secondes entre warnings
        
        # Colors
        self.colors = {
            'bg_dark': '#1a1a1a',
            'bg_card': '#2d2d2d',
            'primary': '#00bcd4',
            'success': '#4caf50',
            'warning': '#ff9800',
            'danger': '#f44336',
            'text': '#ffffff',
            'text_dim': '#b0b0b0'
        }
        
        self.create_ui()
        
    def create_ui(self):
        """Crée l'interface utilisateur"""
        # Header
        header = tk.Frame(self.root, bg=self.colors['primary'], height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="YAQZA AI - ALERT BEFORE DANGER",
            font=('Segoe UI', 22, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        ).pack(pady=15)
        
        # Main container
        main = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Left panel - Video
        left = tk.Frame(main, bg=self.colors['bg_dark'])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        tk.Label(
            left,
            text="CAMERA - FATIGUE DETECTION",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['text']
        ).pack(pady=(0, 5))
        
        video_frame = tk.Frame(left, bg='black', relief=tk.SUNKEN, bd=2)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(
            video_frame,
            bg='black',
            text="Camera not started",
            fg='white',
            font=('Segoe UI', 14)
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Metrics & Analysis
        right = tk.Frame(main, bg=self.colors['bg_dark'], width=600)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right.pack_propagate(False)
        
        # Warning banner
        self.warning_banner = tk.Frame(right, bg=self.colors['danger'], height=60)
        self.warning_banner.pack(fill=tk.X, pady=(0, 10))
        self.warning_banner.pack_propagate(False)
        
        self.warning_label = tk.Label(
            self.warning_banner,
            text="",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['danger'],
            fg='white',
            wraplength=550
        )
        self.warning_label.pack(expand=True)
        self.warning_banner.pack_forget()  # Caché par défaut
        
        # Metrics grid
        metrics = tk.Frame(right, bg=self.colors['bg_dark'])
        metrics.pack(fill=tk.X, pady=(0, 10))
        
        self.fatigue_card = self.create_metric_card(metrics, "FATIGUE", "0", "/100", self.colors['success'])
        self.fatigue_card.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        self.eco_card = self.create_metric_card(metrics, "ECO-SCORE", "100", "/100", self.colors['success'])
        self.eco_card.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        
        self.speed_card = self.create_metric_card(metrics, "VITESSE", "0", "km/h", self.colors['primary'])
        self.speed_card.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        
        self.co2_card = self.create_metric_card(metrics, "CO2", "0", "g/km", self.colors['warning'])
        self.co2_card.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
        
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        
        # Status panel
        self.create_status_panel(right)
        
        # Behavior Analysis panel
        self.create_behavior_panel(right)
        
        # Events log
        self.create_events_panel(right)
        
        # Controls
        self.create_controls(right)
        
        # Footer
        footer = tk.Frame(self.root, bg=self.colors['bg_card'], height=35)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        
        tk.Label(
            footer,
            text="Hack For Good 4.0 | © 2025 Yaqza AI Team",
            font=('Segoe UI', 9),
            bg=self.colors['bg_card'],
            fg=self.colors['text_dim']
        ).pack(pady=8)
        
    def create_metric_card(self, parent, title, value, unit, color):
        """Crée une carte métrique"""
        card = tk.Frame(parent, bg=self.colors['bg_card'], relief=tk.RAISED, bd=1)
        
        tk.Label(
            card,
            text=title,
            font=('Segoe UI', 9),
            bg=self.colors['bg_card'],
            fg=self.colors['text_dim']
        ).pack(pady=(8, 2))
        
        value_frame = tk.Frame(card, bg=self.colors['bg_card'])
        value_frame.pack()
        
        value_label = tk.Label(
            value_frame,
            text=value,
            font=('Segoe UI', 26, 'bold'),
            bg=self.colors['bg_card'],
            fg=color
        )
        value_label.pack(side=tk.LEFT)
        
        tk.Label(
            value_frame,
            text=unit,
            font=('Segoe UI', 10),
            bg=self.colors['bg_card'],
            fg=self.colors['text_dim']
        ).pack(side=tk.LEFT, padx=(3, 0), pady=(8, 0))
        
        card.value_label = value_label
        return card
        
    def create_status_panel(self, parent):
        """Panneau de statut"""
        frame = tk.LabelFrame(
            parent,
            text="SESSION STATUS",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            bd=1
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        
        inner = tk.Frame(frame, bg=self.colors['bg_card'])
        inner.pack(fill=tk.BOTH, padx=8, pady=8)
        
        self.session_label = tk.Label(
            inner,
            text="Session: Not started",
            font=('Segoe UI', 9),
            bg=self.colors['bg_card'],
            fg=self.colors['text'],
            anchor='w'
        )
        self.session_label.pack(fill=tk.X, pady=1)
        
        self.duration_label = tk.Label(
            inner,
            text="Duration: 00:00:00",
            font=('Segoe UI', 9),
            bg=self.colors['bg_card'],
            fg=self.colors['text'],
            anchor='w'
        )
        self.duration_label.pack(fill=tk.X, pady=1)
        
        self.distance_label = tk.Label(
            inner,
            text="Distance: 0.00 km",
            font=('Segoe UI', 9),
            bg=self.colors['bg_card'],
            fg=self.colors['text'],
            anchor='w'
        )
        self.distance_label.pack(fill=tk.X, pady=1)
        
    def create_behavior_panel(self, parent):
        """Panneau d'analyse comportementale"""
        frame = tk.LabelFrame(
            parent,
            text="BEHAVIOR ANALYSIS",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            bd=1
        )
        frame.pack(fill=tk.X, pady=(0, 10))
        
        inner = tk.Frame(frame, bg=self.colors['bg_card'])
        inner.pack(fill=tk.BOTH, padx=8, pady=8)
        
        self.event_labels = {}
        events = [
            ("harsh_acceleration", "Harsh Accelerations"),
            ("harsh_braking", "Harsh Braking"),
            ("speeding", "Speeding"),
            ("idling", "Idling"),
            ("steering_oscillation", "Steering Oscillations")
        ]
        
        for key, label in events:
            lbl = tk.Label(
                inner,
                text=f"{label}: 0",
                font=('Segoe UI', 9),
                bg=self.colors['bg_card'],
                fg=self.colors['text'],
                anchor='w'
            )
            lbl.pack(fill=tk.X, pady=1)
            self.event_labels[key] = lbl
            
    def create_events_panel(self, parent):
        """Panneau d'événements"""
        frame = tk.LabelFrame(
            parent,
            text="EVENTS LOG",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            bd=1
        )
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.events_listbox = tk.Listbox(
            frame,
            font=('Consolas', 8),
            bg=self.colors['bg_card'],
            fg=self.colors['text'],
            yscrollcommand=scrollbar.set,
            selectbackground=self.colors['primary'],
            height=8
        )
        self.events_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.events_listbox.yview)
        
    def create_controls(self, parent):
        """Panneau de contrôle"""
        frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        frame.pack(fill=tk.X)
        
        # Mode selector
        mode_frame = tk.Frame(frame, bg=self.colors['bg_dark'])
        mode_frame.pack(pady=(0, 8))
        
        tk.Label(
            mode_frame,
            text="Driving Mode:",
            font=('Segoe UI', 9),
            bg=self.colors['bg_dark'],
            fg=self.colors['text']
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.mode_var = tk.StringVar(value='mixed')
        mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=['highway', 'urban', 'mixed'],
            state='readonly',
            width=12,
            font=('Segoe UI', 9)
        )
        mode_combo.pack(side=tk.LEFT)
        
        # Buttons
        button_frame = tk.Frame(frame, bg=self.colors['bg_dark'])
        button_frame.pack()
        
        self.start_button = tk.Button(
            button_frame,
            text="START SESSION",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['success'],
            fg='white',
            activebackground='#45a049',
            command=self.start_session,
            width=20,
            height=2,
            cursor='hand2',
            relief=tk.FLAT
        )
        self.start_button.pack(side=tk.LEFT, padx=3)
        
        self.stop_button = tk.Button(
            button_frame,
            text="STOP SESSION",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['danger'],
            fg='white',
            activebackground='#da190b',
            command=self.stop_session,
            width=20,
            height=2,
            state=tk.DISABLED,
            cursor='hand2',
            relief=tk.FLAT
        )
        self.stop_button.pack(side=tk.LEFT, padx=3)
        
    def start_session(self):
        """Démarre une session"""
        if self.is_running:
            return
            
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Init camera
        try:
            predictor_path = os.path.join(
                os.path.dirname(__file__),
                'Fatigue detection',
                'data',
                'shape_predictor_68_face_landmarks.dat'
            )
            
            if os.path.exists(predictor_path):
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.detector = dlib.get_frontal_face_detector()
                    self.predictor = dlib.shape_predictor(predictor_path)
                    self.use_camera = True
                    self.add_event("INFO", "Camera-based fatigue detection activated")
                else:
                    self.use_camera = False
                    self.add_event("WARNING", "Camera not available - simulation mode")
            else:
                self.use_camera = False
                self.add_event("WARNING", "Facial model not found - simulation mode")
        except Exception as e:
            self.use_camera = False
            self.add_event("ERROR", f"Camera error: {str(e)[:50]}")
        
        # Init components
        mode = self.mode_var.get()
        self.simulator = DrivingSimulator(mode=mode)
        self.analyzer = BehaviorAnalyzer()
        self.logger = SessionLogger()
        self.start_time = time.time()
        
        # Reset
        self.yawn_count = 0
        self.yawn_frame_counter = 0
        self.fatigue_score_smoothed = 0.0
        self.queue = [0] * 60
        self.closed_streak = 0
        
        self.session_label.config(text=f"Session: {self.logger.session_id}")
        self.add_event("INFO", f"Session started - Mode: {mode}")
        
        # Start threads
        if self.use_camera:
            self.camera_thread = threading.Thread(target=self.run_camera, daemon=True)
            self.camera_thread.start()
            
        self.session_thread = threading.Thread(target=self.run_monitoring, daemon=True)
        self.session_thread.start()
        
        self.warning_thread = threading.Thread(target=self.run_warning_system, daemon=True)
        self.warning_thread.start()
        
    def stop_session(self):
        """Arrête la session"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.hide_warning()
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        if self.logger:
            summary = self.logger.close()
            self.add_event("INFO", f"Session saved: {self.logger.csv_path}")
            
            kpis = summary['kpis']
            msg = f"""Session Complete!

Duration: {kpis['duration_minutes']:.1f} min
Distance: {kpis['distance_km']} km
Max Fatigue: {kpis['max_fatigue_score']}/100
Eco-Score: {kpis['eco_score']}/100
Total Events: {kpis['total_events']}

Files saved:
- {self.logger.csv_path}
- {self.logger.summary_path}
"""
            messagebox.showinfo("Session Complete", msg)
            
    def run_camera(self):
        """Thread caméra"""
        head_down_streak = 0
        
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 1)
            
            if len(rects) == 0:
                # No face detected
                cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            for rect in rects:
                x, y, w, h = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                landmarks = self.predictor(gray, rect)
                landmarks = landmarks_to_np(landmarks)
                
                # Draw landmarks
                for (lx, ly) in landmarks:
                    cv2.circle(frame, (lx, ly), 1, (0, 0, 255), -1)
                
                # Calculate ratios
                eye_ratio = _eye_ratio(landmarks)
                mouth_ratio = _mouth_ratio(landmarks)
                
                # Eye closure detection
                eye_closed = int(eye_ratio < 0.22)
                self.queue = self.queue[1:] + [eye_closed]
                self.closed_streak = self.closed_streak + 1 if eye_closed else 0
                
                # Yawn detection
                if mouth_ratio > 0.6:
                    self.yawn_frame_counter += 1
                else:
                    if self.yawn_frame_counter >= 15:
                        self.yawn_count += 1
                    self.yawn_frame_counter = 0
                
                # Head pose
                model_points = np.array([
                    (0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
                    (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
                    (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
                ], dtype=np.float64)
                
                image_points = np.array([
                    landmarks[30], landmarks[8], landmarks[36],
                    landmarks[45], landmarks[48], landmarks[54]
                ], dtype=np.float64)
                
                h_img, w_img = frame.shape[:2]
                camera_matrix = np.array([
                    [w_img, 0, w_img/2],
                    [0, w_img, h_img/2],
                    [0, 0, 1]
                ], dtype=np.float64)
                
                pitch_deg = yaw_deg = roll_deg = 0
                try:
                    success, rotation_vector, _ = cv2.solvePnP(
                        model_points, image_points, camera_matrix, np.zeros((4, 1))
                    )
                    if success:
                        rmat, _ = cv2.Rodrigues(rotation_vector)
                        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
                        yaw = np.arctan2(-rmat[2, 0], np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2))
                        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
                        pitch_deg = np.degrees(pitch)
                        yaw_deg = np.degrees(yaw)
                        roll_deg = np.degrees(roll)
                except:
                    pass
                
                self.current_fatigue['head_pitch'] = pitch_deg
                self.current_fatigue['head_yaw'] = yaw_deg
                self.current_fatigue['head_roll'] = roll_deg
                
                # Composite score
                fatigued_eye = (self.closed_streak >= 45 or (sum(self.queue)/len(self.queue)) >= 0.7)
                head_down = pitch_deg > 15.0
                head_down_streak = head_down_streak + 1 if head_down else 0
                fatigued_angle = head_down_streak >= 45
                
                eye_component = min(1.0, sum(self.queue)/len(self.queue))
                yawn_component = min(1.0, self.yawn_count / max(1, 3))
                angle_component = 1.0 if fatigued_angle else (head_down_streak / 45)
                
                raw_score = 0.4 * eye_component + 0.3 * yawn_component + 0.3 * angle_component
                self.fatigue_score_smoothed = 0.9 * self.fatigue_score_smoothed + 0.1 * raw_score
                
                self.current_fatigue['composite_score'] = self.fatigue_score_smoothed * 100
                self.current_fatigue['eye_ratio'] = eye_ratio
                self.current_fatigue['yawn_count'] = self.yawn_count
                
                # Display on frame
                status = "DANGER!" if self.current_fatigue['composite_score'] >= 70 else "SAFE"
                color = (0, 0, 255) if status == "DANGER!" else (0, 255, 0)
                
                cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Fatigue: {int(self.current_fatigue['composite_score'])}/100",
                           (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Yawns: {self.yawn_count}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, f"EAR: {eye_ratio:.2f} MAR: {mouth_ratio:.2f}",
                           (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                break
            
            # Convert for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((720, 540), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.root.after(0, self.update_video, imgtk)
            time.sleep(0.033)
            
    def update_video(self, imgtk):
        """Met à jour le flux vidéo"""
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text="")
        
    def run_monitoring(self):
        """Thread de monitoring"""
        while self.is_running:
            elapsed = time.time() - self.start_time
            
            # Use camera or simulate
            if not self.use_camera:
                time_factor = elapsed / 300
                self.current_fatigue['composite_score'] = min(100, time_factor ** 1.5 * 120)
                self.current_fatigue['yawn_count'] = int(self.current_fatigue['composite_score'] / 20)
            
            fatigue_score = self.current_fatigue['composite_score']
            
            # Generate driving data
            driving_data = self.simulator.update(fatigue_score=fatigue_score)
            
            # Analyze behavior
            events, metrics = self.analyzer.analyze_datapoint(driving_data)
            eco_score = self.analyzer.calculate_eco_score()
            
            # Log
            analysis = {
                'events': events,
                'total_distance_km': metrics['total_distance_km'],
                'eco_score': eco_score,
                'recommendation': ''
            }
            
            self.logger.log_datapoint(self.current_fatigue, driving_data, analysis)
            
            # Update UI
            self.root.after(0, self.update_interface,
                          fatigue_score, eco_score, driving_data, events, metrics, elapsed)
            
            time.sleep(1)
            
    def run_warning_system(self):
        """Thread de système d'alerte"""
        while self.is_running:
            current_time = time.time()
            
            # Check if cooldown passed
            if current_time - self.last_warning_time < self.warning_cooldown:
                time.sleep(1)
                continue
            
            fatigue = self.current_fatigue['composite_score']
            
            # Critical fatigue
            if fatigue >= 70:
                self.root.after(0, self.show_warning,
                              "CRITICAL FATIGUE DETECTED! STOP IMMEDIATELY!",
                              self.colors['danger'])
                self.last_warning_time = current_time
                time.sleep(1)
                continue
            
            # High fatigue
            if fatigue >= 40:
                self.root.after(0, self.show_warning,
                              "WARNING: High fatigue level. Plan a break soon.",
                              self.colors['warning'])
                self.last_warning_time = current_time
                time.sleep(1)
                continue
            
            # Check behavior events
            if self.analyzer:
                recent_harsh = (
                    self.analyzer.event_counts.get('harsh_acceleration', 0) +
                    self.analyzer.event_counts.get('harsh_braking', 0)
                )
                
                if recent_harsh > 10:
                    self.root.after(0, self.show_warning,
                                  f"CAUTION: {recent_harsh} harsh driving events detected!",
                                  self.colors['warning'])
                    self.last_warning_time = current_time
                    
            time.sleep(1)
            
    def show_warning(self, message, color):
        """Affiche un warning"""
        self.warning_banner.config(bg=color)
        self.warning_label.config(text=message, bg=color)
        self.warning_banner.pack(fill=tk.X, pady=(0, 10), before=self.fatigue_card.master)
        
    def hide_warning(self):
        """Cache le warning"""
        self.warning_banner.pack_forget()
        
    def update_interface(self, fatigue, eco_score, driving, events, metrics, elapsed):
        """Met à jour l'interface"""
        # Update metrics
        fatigue_color = self.get_color(fatigue, 40, 70, True)
        self.fatigue_card.value_label.config(text=str(int(fatigue)), fg=fatigue_color)
        
        eco_color = self.get_color(eco_score, 60, 80, False)
        self.eco_card.value_label.config(text=str(int(eco_score)), fg=eco_color)
        
        self.speed_card.value_label.config(text=str(int(driving['speed_kmh'])))
        self.co2_card.value_label.config(text=str(int(driving['co2_estimate_gkm'])))
        
        # Update status
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        self.duration_label.config(text=f"Duration: {h:02d}:{m:02d}:{s:02d}")
        self.distance_label.config(text=f"Distance: {metrics['total_distance_km']:.2f} km")
        
        # Update behavior counters
        for key, lbl in self.event_labels.items():
            count = self.analyzer.event_counts.get(key, 0)
            label_text = lbl.cget('text').split(':')[0]
            lbl.config(text=f"{label_text}: {count}")
        
        # Log events
        if events:
            for event in events:
                level = "ALERT" if event in ['harsh_acceleration', 'harsh_braking', 'speeding'] else "INFO"
                self.add_event(level, self.format_event(event, driving))
        
    def get_color(self, value, threshold1, threshold2, inverse=False):
        """Retourne la couleur selon le seuil"""
        if inverse:  # Higher is worse
            if value < threshold1:
                return self.colors['success']
            elif value < threshold2:
                return self.colors['warning']
            else:
                return self.colors['danger']
        else:  # Lower is worse
            if value >= threshold2:
                return self.colors['success']
            elif value >= threshold1:
                return self.colors['warning']
            else:
                return self.colors['danger']
                
    def format_event(self, event, driving):
        """Formate un événement"""
        msgs = {
            'harsh_acceleration': f"Harsh acceleration ({driving['acceleration_ms2']:.1f} m/s²)",
            'harsh_braking': f"Harsh braking ({driving['acceleration_ms2']:.1f} m/s²)",
            'speeding': f"Speeding ({driving['speed_kmh']:.0f} km/h)",
            'idling': "Prolonged idling detected",
            'steering_oscillation': f"Steering oscillation ({driving['steering_angle_deg']:.1f}°)"
        }
        return msgs.get(event, event)
        
    def add_event(self, level, message):
        """Ajoute un événement au log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        text = f"[{timestamp}] [{level}] {message}"
        
        self.events_listbox.insert(0, text)
        
        if self.events_listbox.size() > 100:
            self.events_listbox.delete(tk.END)
        
        # Color coding
        if level == "ERROR" or level == "CRITICAL":
            self.events_listbox.itemconfig(0, fg='#ff5555')
        elif level == "ALERT" or level == "WARNING":
            self.events_listbox.itemconfig(0, fg='#ffaa00')
        else:
            self.events_listbox.itemconfig(0, fg='#88ff88')


def main():
    root = tk.Tk()
    app = YaqzaAIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
