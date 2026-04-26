import cv2
import numpy as np
import os

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
            cv2.putText(frame, "Number Plate", (x, y - 5), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        return frame

    def extract_roi(self, frame, detection):
        """Extracts the Region of Interest (ROI) for the number plate."""
        (x, y, w, h) = detection
        return frame[y:y+h, x:x+w]

class VehicleCounter:
    """Background subtraction based vehicle counting system."""
    def __init__(self, line_position=550, min_width=80, min_height=80):
        self.line_position = line_position
        self.min_width = min_width
        self.min_height = min_height
        self.subtractor = cv2.createBackgroundSubtractorMOG2()
        self.detect_list = []
        self.counter = 0
        self.offset = 6

    def process_frame(self, frame):
        """Processes frame for vehicle detection and counting."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 5)
        img_sub = self.subtractor.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        processed_frame = frame.copy()
        cv2.line(processed_frame, (25, self.line_position), (1200, self.line_position), (225, 127, 0), 3)
        
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if (w >= self.min_width) and (h >= self.min_height):
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                cx = x + int(w / 2)
                cy = y + int(h / 2)
                self.detect_list.append((cx, cy))
                cv2.circle(processed_frame, (cx, cy), 4, (0, 0, 255), -1)
        
        # Update counter
        for (cx, cy) in self.detect_list[:]:
            if (self.line_position - self.offset) < cy < (self.line_position + self.offset):
                self.counter += 1
                cv2.line(processed_frame, (25, self.line_position), (1200, self.line_position), (0, 127, 255), 3)
                self.detect_list.remove((cx, cy))
                
        cv2.putText(processed_frame, f"Vehicle Count: {self.counter}", (550, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 225), 5)
        
        return processed_frame
