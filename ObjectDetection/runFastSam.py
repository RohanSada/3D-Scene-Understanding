import cv2
from ultralytics import FastSAM
import torch

class FastSam():
    def __init__(self, model_path='./models/FastSAM-s.pt', conf_threshold=0.4, iou_threshold=0.9):
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print(f"Success: Running on Apple Silicon GPU ({self.device})")
        else:
            self.device = 'cpu'
            print("Warning: MPS not found. Running on CPU (will be slower).")
        self.model = FastSAM(model_path)
        self.conf_threshold=conf_threshold
        self.iou_threshold=iou_threshold
    
    def run_sam(self, frame, imgsz=320, retina_masks=True):
        results = self.model(
            frame, 
            device=self.device, 
            retina_masks=retina_masks, 
            imgsz=imgsz, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold,
            verbose=False 
        )
        return results
    
    def visualise_sam_results(self, frame, results):
        annotated_frame = results[0].cpu().plot()
        return annotated_frame



if __name__=='__main__':
    video_path = '../Videos/demo.mp4'
    model_path = './models/FastSAM-s.pt'
    skip_frames = 10
    frame_count = 0
    fastsam = FastSam(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Video not found')
        exit()
    
    last_annotated_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            should_run = (frame_count % skip_frames == 0)
            
            if should_run == True:
                results = fastsam.run_sam(frame)
                annotated_frame = fastsam.visualise_sam_results(frame, results)
                last_annotated_frame = annotated_frame
            else:
                annotated_frame = last_annotated_frame
            
            cv2.imshow("FastSAM Output", annotated_frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break         
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

