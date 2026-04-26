import cv2
import os
import time
from core.detector import PlateDetector, VehicleTracker

def main():
    # Configuration
    SOURCE = 'car.mp4'  # Change to 0 for webcam
    SPEED_LIMIT = 60    # Set speed limit in km/h
    OUTPUT_DIR = 'exports/overspeeding'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize components
    plate_detector = PlateDetector()
    vehicle_tracker = VehicleTracker(pixels_per_meter=15) # Calibrate pixels_per_meter for your video
    
    cap = cv2.VideoCapture(SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    
    print(f"Monitoring Started. Speed Limit: {SPEED_LIMIT} km/h")
    print("Saving plates of vehicles exceeding the limit...")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 1. Track vehicles and calculate speed
        frame = vehicle_tracker.process_frame(frame, fps)
        
        # 2. Check for overspeeding vehicles
        for vid, data in vehicle_tracker.tracked_vehicles.items():
            speed = data['speed']
            bbox = data['bbox']
            
            if speed > SPEED_LIMIT:
                # 3. Detect number plate ONLY for overspeeding vehicles
                # We crop the frame to the vehicle's bounding box to increase accuracy and speed
                vehicle_roi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                
                if vehicle_roi.size > 0:
                    plates = plate_detector.detect(vehicle_roi)
                    if plates:
                        # Extract the plate from the ROI (adjust coordinates back to original frame)
                        px, py, pw, ph = plates[0]
                        plate_img = vehicle_roi[py:py+ph, px:px+pw]
                        
                        # Save overspeeding plate
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        save_path = os.path.join(OUTPUT_DIR, f"speed_{int(speed)}_{timestamp}_{vid}.jpg")
                        cv2.imwrite(save_path, plate_img)
                        
                        # Visual indication on frame
                        cv2.rectangle(frame, (bbox[0]+px, bbox[1]+py), (bbox[0]+px+pw, bbox[1]+py+ph), (0, 0, 255), 3)
                        cv2.putText(frame, "OVERSPEEDING!", (bbox[0], bbox[1]-25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("AI Speed Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
