import cv2
import os
import json
import time
from core.detector import PlateDetector, VehicleTracker, PlateReader

def main():
    # Configuration
    SOURCE = 'car.mp4' 
    SPEED_LIMIT = 60
    OUTPUT_DIR = 'exports/alerts'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Watchlist
    with open('watchlist.json', 'r') as f:
        watchlist = json.load(f)
    
    # Initialize components
    plate_detector = PlateDetector()
    vehicle_tracker = VehicleTracker(pixels_per_meter=15)
    plate_reader = PlateReader()
    
    cap = cv2.VideoCapture(SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    
    print("Surveillance System Active...")
    print(f"Watchlist: {watchlist}")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 1. Vehicle Tracking and Speed
        frame = vehicle_tracker.process_frame(frame, fps)
        
        # 2. Process each tracked vehicle
        for vid, data in vehicle_tracker.tracked_vehicles.items():
            speed = data['speed']
            bbox = data['bbox']
            
            # Logic A: Overspeeding Detection
            is_overspeeding = speed > SPEED_LIMIT
            
            # Logic B: Stolen/Missing Vehicle Search (Always detect plate)
            # To optimize, we can run OCR every 10th frame or only for new vehicles
            
            # We only run OCR if it's overspeeding OR if we want to check against watchlist
            if is_overspeeding or True: # 'True' here means always check watchlist
                vehicle_roi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                if vehicle_roi.size > 0:
                    plates = plate_detector.detect(vehicle_roi)
                    if plates:
                        px, py, pw, ph = plates[0]
                        plate_img = vehicle_roi[py:py+ph, px:px+pw]
                        
                        # READ PLATE TEXT
                        plate_text = plate_reader.read_plate(plate_img)
                        
                        if plate_text:
                            # Check if plate is in watchlist
                            is_stolen = any(wp.upper() in plate_text.upper() for wp in watchlist)
                            
                            # Handle Visuals and Alerts
                            if is_stolen:
                                color = (0, 0, 255) # RED for ALERT
                                label = f"STOLEN: {plate_text}"
                                # Save alert image
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                cv2.imwrite(os.path.join(OUTPUT_DIR, f"alert_{plate_text}_{timestamp}.jpg"), frame)
                            elif is_overspeeding:
                                color = (0, 255, 255) # YELLOW for Speeding
                                label = f"SPEEDING: {plate_text}"
                            else:
                                color = (0, 255, 0) # GREEN for Normal
                                label = f"Plate: {plate_text}"
                                
                            # Draw on original frame
                            # Plate coordinates relative to original frame
                            abs_x = bbox[0] + px
                            abs_y = bbox[1] + py
                            cv2.rectangle(frame, (abs_x, abs_y), (abs_x + pw, abs_y + ph), color, 3)
                            cv2.putText(frame, label, (abs_x, abs_y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Police Surveillance AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
