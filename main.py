import cv2
import os
from core.detector import PlateDetector, VehicleCounter

def main():
    # Configuration
    SOURCE = 0  # Use 0 for webcam, or 'car.mp4' for file
    OUTPUT_DIR = 'exports'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize components
    plate_detector = PlateDetector()
    vehicle_counter = VehicleCounter()
    
    cap = cv2.VideoCapture(SOURCE)
    count = 0

    print("Starting Detection... Press 's' to save, 'q' to quit.")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 1. Vehicle Counting Logic
        frame = vehicle_counter.process_frame(frame)
        
        # 2. Number Plate Detection Logic
        plates = plate_detector.detect(frame)
        frame = plate_detector.draw_detections(frame, plates)
        
        cv2.imshow("Smart Traffic Monitoring", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if plates:
                roi = plate_detector.extract_roi(frame, plates[0])
                save_path = os.path.join(OUTPUT_DIR, f"plate_{count}.jpg")
                cv2.imwrite(save_path, roi)
                print(f"Saved: {save_path}")
                count += 1
        elif key == ord('q') or key == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
