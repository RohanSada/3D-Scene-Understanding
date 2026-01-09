import cv2
import os
from runFastSam import FastSam
from yolo_open_vocab import YoloWorldMac
import numpy as np

class FuseSamYolo():
    def __init__(self):
        self.yolo = YoloWorldMac()
        self.sam = FastSam()

    def _get_iou(self, box, mask_box):
        # Box format: [x1, y1, x2, y2]
        x1 = max(box[0], mask_box[0])
        y1 = max(box[1], mask_box[1])
        x2 = min(box[2], mask_box[2])
        y2 = min(box[3], mask_box[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_mask = (mask_box[2] - mask_box[0]) * (mask_box[3] - mask_box[1])
        
        union = area_box + area_mask - intersection
        return intersection / union if union > 0 else 0

    def process_frame(self, frame):
        yolo_results = self.yolo.get_outputs(frame)
        sam_results = self.sam.run_sam(frame)
        
        fused_results = []

        if sam_results[0].masks is not None:
            sam_boxes = sam_results[0].boxes.xyxy.cpu().numpy()
            sam_masks = sam_results[0].masks.data.cpu().numpy()
            
            yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            yolo_cls = yolo_results[0].boxes.cls.cpu().numpy()
            yolo_conf = yolo_results[0].boxes.conf.cpu().numpy()

            for y_box, cls_id, conf in zip(yolo_boxes, yolo_cls, yolo_conf):
                label_name = yolo_results[0].names[int(cls_id)]                
                best_iou = 0
                best_mask_idx = -1
                
                for i, s_box in enumerate(sam_boxes):
                    iou = self._get_iou(y_box, s_box)
                    if iou > 0.5 and iou > best_iou: 
                        best_iou = iou
                        best_mask_idx = i
                
                if best_mask_idx != -1:
                    result_entry = {
                        "label": label_name,
                        "confidence": float(conf),
                        "box": y_box.tolist(),        # YOLO box [x1, y1, x2, y2]
                        "mask_box": sam_boxes[best_mask_idx].tolist(), # SAM box (often tighter)
                        "mask": sam_masks[best_mask_idx], # Raw boolean/float mask array
                        "iou": float(best_iou)
                    }
                    fused_results.append(result_entry)  
        return fused_results
    
    def display_fused_results(self, frame, fused_results):
        """
        Takes a raw frame and the list of fused results (dictionaries).
        Draws masks, boxes, and labels.
        Returns the annotated frame.
        """
        # Create a copy so we don't mess up the original frame
        annotated_frame = frame.copy()
        
        # Define some colors (B, G, R)
        MASK_COLOR = (0, 255, 0)      # Green for masks
        BOX_COLOR = (0, 255, 0)       # Green for boxes
        TEXT_COLOR = (255, 0, 0)  # White text
        
        for result in fused_results:
            label = result['label']
            conf = result['confidence']
            box = result['box']     # [x1, y1, x2, y2]
            mask = result['mask']   # Raw mask array
            
            # --- 1. DRAW MASK ---
            # Resize mask to match frame dimensions if needed
            if mask.shape[:2] != annotated_frame.shape[:2]:
                mask = cv2.resize(mask, (annotated_frame.shape[1], annotated_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Create a colored overlay
            # We create a boolean mask where the object is
            mask_bool = mask.astype(bool)
            
            # Apply color only to the masked region
            # Create a color layer
            color_layer = np.zeros_like(annotated_frame, dtype=np.uint8)
            color_layer[mask_bool] = MASK_COLOR
            
            # Blend the color layer with the original frame (Alpha Blending)
            # 0.6 = Original Image Strength, 0.4 = Mask Color Strength
            alpha = 0.4
            
            # We only want to blend where the mask is, to save processing time
            # Extract the region of interest (ROI) from the image
            roi = annotated_frame[mask_bool]
            
            # Blend manually
            blended_roi = (roi * (1 - alpha) + np.array(MASK_COLOR) * alpha).astype(np.uint8)
            
            # Put it back
            annotated_frame[mask_bool] = blended_roi

            # --- 2. DRAW BOX ---
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            
            # --- 3. DRAW LABEL ---
            label_text = f"{label} {conf:.2f}"
            
            # Get text size for a nice background box
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Draw text background
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), BOX_COLOR, -1)
            
            # Draw text
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
                        
        return annotated_frame
                    

if __name__=='__main__':
    video_path = '../Videos/demo.mp4'
    skip_frames = 10
    frame_count = 0
    fusedata = FuseSamYolo()

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
                fused_results = fusedata.process_frame(frame)
                annotated_frame = fusedata.display_fused_results(frame, fused_results)
                last_annotated_frame = annotated_frame
            else:
                annotated_frame = last_annotated_frame
            
            cv2.imshow("Fused Data Output", annotated_frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break         
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
