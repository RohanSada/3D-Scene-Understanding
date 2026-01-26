import cv2
from ultralytics import FastSAM
import torch
import time

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
    
    def run_sam(self, frame, imgsz=1024, retina_masks=True):
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
    
    def visualise_sam_results(self, results):
        annotated_frame = results[0].cpu().plot()
        return annotated_frame



if __name__=='__main__':
    frame_path = '../Videos/Frames/frame_00320.jpg'
    model_path = './models/FastSAM-s.pt'

    sam_engine = FastSam(model_path=model_path, conf_threshold=0.6, iou_threshold=0.9)

    frame = cv2.imread(frame_path)
    for i in range(10):
        sam_results = sam_engine.run_sam(frame)
    t0 = time.time()
    sam_results = sam_engine.run_sam(frame)
    print("SAM Inference Time: ", time.time()-t0)
    annotated_sam_output = sam_engine.visualise_sam_results(sam_results)

    cv2.imshow("SAM Output", annotated_sam_output)
    cv2.waitKey(0)
    