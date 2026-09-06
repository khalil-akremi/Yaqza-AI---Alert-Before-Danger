# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 22:20:37 2018

@author: James Wu
"""

import os
import dlib
import cv2
import numpy as np

        
def landmarks_to_np(landmarks, dtype="int"):
    # 获取landmarks的数量
    num = landmarks.num_parts
    
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)
    
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

#==============================================================================
#   **************************主函数入口***********************************
#==============================================================================

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
    # Use outer lip landmarks: vertical distances averaged over three pairs normalized by horizontal width.
    # Pairs: (51,59), (53,57), (50,58); horizontal reference (48,54)
    v1 = np.linalg.norm(landmarks[51] - landmarks[59])
    v2 = np.linalg.norm(landmarks[53] - landmarks[57])
    v3 = np.linalg.norm(landmarks[50] - landmarks[58])
    vertical_mean = (v1 + v2 + v3) / 3.0
    horizontal = np.linalg.norm(landmarks[48] - landmarks[54])
    return vertical_mean / horizontal if horizontal else 0.0

def main(
    camera_index: int = 0,
    queue_length: int = 60,
    eye_closed_threshold: float = 0.22,
    mouth_open_threshold: float = 0.6,
    yawn_min_frames: int = 15,
    yawn_fatigue_threshold: int = 3,
    min_closed_streak_for_fatigue: int = 45,
    queue_ratio_fatigue_threshold: float = 0.7,
    pitch_down_threshold_deg: float = 15.0,
    head_down_min_frames: int = 45,
    eye_weight: float = 0.4,
    yawn_weight: float = 0.3,
    angle_weight: float = 0.3
):
    predictor_path = os.path.join(os.path.dirname(__file__), 'data', 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Missing landmark model at {predictor_path}. Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and decompress into the data folder.")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    queue = [0] * queue_length
    frame_counter = 0
    yawn_frame_counter = 0
    yawn_count = 0
    closed_streak = 0
    head_down_streak = 0
    fatigue_score_smoothed = 0.0

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for i, rect in enumerate(rects):
            x = rect.left(); y = rect.top(); w = rect.right() - x; h = rect.bottom() - y
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"Face #{i+1}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            landmarks = predictor(gray, rect)
            landmarks = landmarks_to_np(landmarks)
            for (lx, ly) in landmarks:
                cv2.circle(img, (lx, ly), 2, (0, 0, 255), -1)

            eye_ratio = _eye_ratio(landmarks)
            mouth_ratio = _mouth_ratio(landmarks)

            eye_closed_flag = int(eye_ratio < eye_closed_threshold)
            queue = queue[1:] + [eye_closed_flag]
            closed_streak = closed_streak + 1 if eye_closed_flag else 0

            # Yawn detection logic: count consecutive frames of mouth open
            if mouth_ratio > mouth_open_threshold:
                yawn_frame_counter += 1
            else:
                if yawn_frame_counter >= yawn_min_frames:
                    yawn_count += 1
                yawn_frame_counter = 0

            fatigued_eye = (
                closed_streak >= min_closed_streak_for_fatigue or
                (sum(queue) / len(queue)) >= queue_ratio_fatigue_threshold
            )
            fatigued_yawn = yawn_count >= yawn_fatigue_threshold

            # Head pose estimation for angle fatigue
            # 3D model points (approx units in arbitrary scale)
            model_points = np.array([
                (0.0, 0.0, 0.0),            # Nose tip 30
                (0.0, -63.6, -12.5),        # Chin 8
                (-43.3, 32.7, -26.0),       # Left eye left corner 36
                (43.3, 32.7, -26.0),        # Right eye right corner 45
                (-28.9, -28.9, -24.1),      # Left Mouth corner 48
                (28.9, -28.9, -24.1)        # Right mouth corner 54
            ], dtype=np.float64)
            image_points = np.array([
                landmarks[30],  # Nose tip
                landmarks[8],   # Chin
                landmarks[36],  # Left eye left corner
                landmarks[45],  # Right eye right corner
                landmarks[48],  # Left Mouth corner
                landmarks[54]   # Right mouth corner
            ], dtype=np.float64)

            h, w = img.shape[:2]
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            pitch_deg = 0.0
            try:
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    rmat, _ = cv2.Rodrigues(rotation_vector)
                    sy = np.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])
                    singular = sy < 1e-6
                    if not singular:
                        pitch = np.arctan2(rmat[2,1], rmat[2,2])
                        yaw = np.arctan2(-rmat[2,0], sy)
                        roll = np.arctan2(rmat[1,0], rmat[0,0])
                    else:
                        pitch = np.arctan2(-rmat[1,2], rmat[1,1])
                        yaw = np.arctan2(-rmat[2,0], sy)
                        roll = 0
                    # Convert to degrees
                    pitch_deg = np.degrees(pitch)
                    yaw_deg = np.degrees(yaw)
                    roll_deg = np.degrees(roll)
                else:
                    yaw_deg = roll_deg = 0.0
            except Exception:
                yaw_deg = roll_deg = 0.0

            # Consider downward pitch (positive angle means face down depending on coordinate; adjust sign if needed)
            head_down = pitch_deg > pitch_down_threshold_deg
            head_down_streak = head_down_streak + 1 if head_down else 0
            fatigued_angle = head_down_streak >= head_down_min_frames

            # Composite fatigue score
            eye_component = min(1.0, (sum(queue)/len(queue)))
            yawn_component = min(1.0, yawn_count / max(1, yawn_fatigue_threshold))
            angle_component = 1.0 if fatigued_angle else (head_down_streak / head_down_min_frames)
            raw_score = eye_weight*eye_component + yawn_weight*yawn_component + angle_weight*angle_component
            fatigue_score_smoothed = 0.9 * fatigue_score_smoothed + 0.1 * raw_score
            fatigue_score_percent = int(fatigue_score_smoothed * 100)

            status_text = "WARNING !" if (fatigued_eye or fatigued_yawn or fatigued_angle or fatigue_score_percent >= 70) else "SAFE"
            status_color = (0, 0, 255) if status_text.startswith("WARN") else (0, 255, 0)
            cv2.putText(img, status_text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
            cv2.putText(img, f"Score: {fatigue_score_percent}", (100, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(img, f"YawnCount: {yawn_count}", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"EyeRatio: {eye_ratio:.2f}", (100, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(img, f"MouthRatio: {mouth_ratio:.2f}", (100, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(img, f"Pitch: {pitch_deg:.1f}°", (100, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(img, f"Yaw: {yaw_deg:.1f}° Roll: {roll_deg:.1f}°", (100, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

            frame_counter += 1
            if frame_counter % 120 == 0:
                print(
                    f"EyeRatio={eye_ratio:.3f} MouthRatio={mouth_ratio:.3f} Pitch={pitch_deg:.2f} Yawns={yawn_count} Score={fatigue_score_percent} Status={status_text}"
                )

        cv2.imshow("Result", img)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:  # Esc
            break

    cap.release()
    cv2.destroyAllWindows()


class FatigueDetector:
    """
    Thread-safe wrapper around the real fatigue detection logic.
    Used by integrated_demo.py to read fatigue metrics from a background thread.
    """

    def __init__(
        self,
        camera_index: int = 0,
        queue_length: int = 60,
        eye_closed_threshold: float = 0.22,
        mouth_open_threshold: float = 0.6,
        yawn_min_frames: int = 15,
        yawn_fatigue_threshold: int = 3,
        min_closed_streak_for_fatigue: int = 45,
        queue_ratio_fatigue_threshold: float = 0.7,
        pitch_down_threshold_deg: float = 15.0,
        head_down_min_frames: int = 45,
        eye_weight: float = 0.4,
        yawn_weight: float = 0.3,
        angle_weight: float = 0.3,
    ):
        self.camera_index = camera_index
        self.queue_length = queue_length
        self.eye_closed_threshold = eye_closed_threshold
        self.mouth_open_threshold = mouth_open_threshold
        self.yawn_min_frames = yawn_min_frames
        self.yawn_fatigue_threshold = yawn_fatigue_threshold
        self.min_closed_streak_for_fatigue = min_closed_streak_for_fatigue
        self.queue_ratio_fatigue_threshold = queue_ratio_fatigue_threshold
        self.pitch_down_threshold_deg = pitch_down_threshold_deg
        self.head_down_min_frames = head_down_min_frames
        self.eye_weight = eye_weight
        self.yawn_weight = yawn_weight
        self.angle_weight = angle_weight

        # Shared metrics dict updated by the capture loop
        import threading
        self._lock = threading.Lock()
        self._metrics = {
            'composite_score': 0,
            'eye_ratio': 0.3,
            'yawn_count': 0,
            'head_pitch': 0.0,
            'head_yaw': 0.0,
            'head_roll': 0.0,
        }
        self._running = False
        self._cap = None

        predictor_path = os.path.join(os.path.dirname(__file__), 'data', 'shape_predictor_68_face_landmarks.dat')
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"Missing landmark model at {predictor_path}. "
                "Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
                "and decompress into the data/ folder."
            )
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(predictor_path)

    def get_metrics(self) -> dict:
        """Return a copy of the latest fatigue metrics (thread-safe)."""
        with self._lock:
            return dict(self._metrics)

    def start(self):
        """Open the camera and start the background processing loop."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")
        self._running = True
        import threading
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the processing loop and release the camera."""
        self._running = False
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()

    def _loop(self):
        queue = [0] * self.queue_length
        yawn_frame_counter = 0
        yawn_count = 0
        closed_streak = 0
        head_down_streak = 0
        fatigue_score_smoothed = 0.0

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ], dtype=np.float64)

        while self._running and self._cap.isOpened():
            ret, img = self._cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = self._detector(gray, 1)

            pitch_deg = yaw_deg = roll_deg = 0.0
            eye_ratio_val = 0.3

            for rect in rects:
                x = rect.left(); y = rect.top(); w = rect.right() - x; h_r = rect.bottom() - y
                cv2.rectangle(img, (x, y), (x + w, y + h_r), (0, 255, 0), 2)

                landmarks = self._predictor(gray, rect)
                landmarks = landmarks_to_np(landmarks)

                eye_ratio_val = _eye_ratio(landmarks)
                mouth_ratio = _mouth_ratio(landmarks)

                eye_closed_flag = int(eye_ratio_val < self.eye_closed_threshold)
                queue = queue[1:] + [eye_closed_flag]
                closed_streak = closed_streak + 1 if eye_closed_flag else 0

                if mouth_ratio > self.mouth_open_threshold:
                    yawn_frame_counter += 1
                else:
                    if yawn_frame_counter >= self.yawn_min_frames:
                        yawn_count += 1
                    yawn_frame_counter = 0

                fatigued_eye = (
                    closed_streak >= self.min_closed_streak_for_fatigue or
                    (sum(queue) / len(queue)) >= self.queue_ratio_fatigue_threshold
                )
                fatigued_yawn = yawn_count >= self.yawn_fatigue_threshold

                h_img, w_img = img.shape[:2]
                focal_length = w_img
                center = (w_img / 2, h_img / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)
                dist_coeffs = np.zeros((4, 1), dtype=np.float64)
                image_points = np.array([
                    landmarks[30], landmarks[8], landmarks[36],
                    landmarks[45], landmarks[48], landmarks[54]
                ], dtype=np.float64)

                try:
                    success, rvec, tvec = cv2.solvePnP(
                        model_points, image_points, camera_matrix, dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if success:
                        rmat, _ = cv2.Rodrigues(rvec)
                        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
                        if sy >= 1e-6:
                            pitch_deg = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
                            yaw_deg = np.degrees(np.arctan2(-rmat[2, 0], sy))
                            roll_deg = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
                        else:
                            pitch_deg = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
                            yaw_deg = np.degrees(np.arctan2(-rmat[2, 0], sy))
                            roll_deg = 0.0
                except Exception:
                    pass

                head_down = pitch_deg > self.pitch_down_threshold_deg
                head_down_streak = head_down_streak + 1 if head_down else 0
                fatigued_angle = head_down_streak >= self.head_down_min_frames

                eye_component = min(1.0, sum(queue) / len(queue))
                yawn_component = min(1.0, yawn_count / max(1, self.yawn_fatigue_threshold))
                angle_component = 1.0 if fatigued_angle else (head_down_streak / self.head_down_min_frames)
                raw_score = (
                    self.eye_weight * eye_component +
                    self.yawn_weight * yawn_component +
                    self.angle_weight * angle_component
                )
                fatigue_score_smoothed = 0.9 * fatigue_score_smoothed + 0.1 * raw_score
                fatigue_score_percent = int(fatigue_score_smoothed * 100)

                status_text = "WARNING!" if (
                    fatigued_eye or fatigued_yawn or fatigued_angle or fatigue_score_percent >= 70
                ) else "SAFE"
                status_color = (0, 0, 255) if "WARN" in status_text else (0, 255, 0)
                cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(img, f"Fatigue: {fatigue_score_percent}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(img, f"Yawns: {yawn_count}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(img, f"Pitch: {pitch_deg:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                with self._lock:
                    self._metrics = {
                        'composite_score': fatigue_score_percent,
                        'eye_ratio': round(eye_ratio_val, 3),
                        'yawn_count': yawn_count,
                        'head_pitch': round(pitch_deg, 1),
                        'head_yaw': round(yaw_deg, 1),
                        'head_roll': round(roll_deg, 1),
                    }

            cv2.imshow("Fatigue Detection", img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC stops the demo
                self._running = False
                break

        self._cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
