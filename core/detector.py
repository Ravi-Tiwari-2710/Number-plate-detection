import cv2
import numpy as np
import os
import time

class PlateDetector:
    """Professional Number Plate Detection system using Haar Cascades."""
    def __init__(self, cascade_path='haarcascade_russian_plate_number.xml', min_area=500):
        self.min_area = min_area
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade file not found at {cascade_path}")
        self.plate_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        """Detects number plates in a given frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(gray, 1.1, 4)
        
        detections = []
        for (x, y, w, h) in plates:
            if (w * h) > self.min_area:
                detections.append((x, y, w, h))
        
        return detections

    def draw_detections(self, frame, detections):
        """Visualizes detected plates on the frame."""
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "OVERSPEEDING PLATE", (x, y - 5), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def extract_roi(self, frame, detection):
        """Extracts the Region of Interest (ROI) for the number plate."""
        (x, y, w, h) = detection
        return frame[y:y+h, x:x+w]

class VehicleTracker:
    """Tracks vehicles and calculates their speed."""
    def __init__(self, line_position=550, min_width=80, min_height=80, pixels_per_meter=10):
        self.line_position = line_position
        self.min_width = min_width
        self.min_height = min_height
        self.pixels_per_meter = pixels_per_meter
        self.subtractor = cv2.createBackgroundSubtractorMOG2()
        
        # { vehicle_id: {'last_pos': (cx, cy), 'last_time': timestamp, 'speed': 0} }
        self.tracked_vehicles = {}
        self.next_id = 0
        self.offset = 6

    def process_frame(self, frame, fps):
        """Processes frame, tracks vehicles, and calculates speed."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 5)
        img_sub = self.subtractor.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        processed_frame = frame.copy()
        cv2.line(processed_frame, (25, self.line_position), (1200, self.line_position), (225, 127, 0), 3)
        
        current_frame_centers = []
        
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if (w >= self.min_width) and (h >= self.min_height):
                cx = x + int(w / 2)
                cy = y + int(h / 2)
                current_frame_centers.append((cx, cy, x, y, w, h))
        
        # Simple Centroid Tracking
        new_tracked_vehicles = {}
        for (cx, cy, x, y, w, h) in current_frame_centers:
            assigned_id = None
            for vid, data in self.tracked_vehicles.items():
                dist = np.hypot(cx - data['last_pos'][0], cy - data['last_pos'][1])
                if dist < 50: # Threshold for same vehicle
                    assigned_id = vid
                    break
            
            if assigned_id is None:
                assigned_id = self.next_id
                self.next_id += 1
            
            # Speed Calculation
            speed = 0
            if assigned_id in self.tracked_vehicles:
                prev_pos = self.tracked_vehicles[assigned_id]['last_pos']
                prev_time = self.tracked_vehicles[assigned_id]['last_time']
                
                # Calculate distance in pixels
                distance_px = np.hypot(cx - prev_pos[0], cy - prev_pos[1])
                distance_m = distance_px / self.pixels_per_meter
                
                time_diff = time.time() - prev_time
                if time_diff > 0:
                    # Speed = distance / time (m/s) -> convert to km/h
                    speed = (distance_m / time_diff) * 3.6
            
            new_tracked_vehicles[assigned_id] = {
                'last_pos': (cx, cy),
                'last_time': time.time(),
                'speed': speed,
                'bbox': (x, y, w, h)
            }
            
            # Draw Vehicle and Speed
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(processed_frame, f"ID:{assigned_id} {int(speed)}km/h", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self.tracked_vehicles = new_tracked_vehicles
        return processed_frame
