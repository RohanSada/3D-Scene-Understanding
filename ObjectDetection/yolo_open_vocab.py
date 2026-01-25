import cv2
import torch
from ultralytics import YOLO
import time
import sys
import argparse
sys.path.append('../')

class YoloWorldMac:
    def __init__(self, model_path='./models/yolov8s-world.pt', classes=None):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.fuse()
        
        if classes:
            self.set_classes(classes)
            
    def set_classes(self, new_classes):
        self.model.set_classes(new_classes)

    def get_outputs(self, frame, imgz=640, conf=0.5):
        # Inference with specific optimizations
        # verbose=False reduces I/O overhead
        # stream=True is more memory efficient for video
        results = self.model.predict(
            frame, 
            device=self.device, 
            conf=conf, 
            verbose=False,
            imgsz=imgz
        )
        return results

    def process_and_display(self, frame, conf=0.25):
        """
        Method 2: Takes a frame, processes output, and displays results on screen.
        Returns the annotated frame for writing to video if needed.
        """
        results = self.get_outputs(frame, conf)
        
        # Plotting happens on CPU usually, so we let Ultralytics handle the transfer internally
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow("YOLO-World M2 Optimized", annotated_frame)
        return annotated_frame

def main():
    video_path = '../Videos/demo.mp4'
    # --- CONFIGURATION ---
    # Define what you want to find (Open Vocabulary magic)
    
    # Initialize Detector
    detector = YoloWorldMac(model_path='./models/yolov8s-world.pt')
    
    # Start Video Capture (0 for webcam)
    cap = cv2.VideoCapture(video_path)
    
    # Optimization: Set camera resolution to reasonable limits (1280x720) 
    # High res inputs kill FPS on inference
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting stream... Press 'q' to quit.")
    
    prev_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- RUN DETECTION ---
            detector.process_and_display(frame)
            
            # --- FPS CALCULATION ---
            # Measure actual loop speed
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            print(f"FPS: {fps:.1f}", end='\r') # Print FPS in place
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCleanup done.")

if __name__ == "__main__":
    main()