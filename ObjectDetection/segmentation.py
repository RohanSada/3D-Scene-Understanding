from ultralytics import YOLO
import cv2
import numpy as np
import sys
import torch


class YOLOSegmentationDisplay:
    def __init__(self, model_path="yolov8n-seg.pt", conf_threshold=0.5):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.model.to(self.device)
        self.model.fuse()

    def segment_frame(self, frame):
        results = self.model.predict(
            frame, device=self.device, conf=self.conf_threshold, verbose=False
        )[0]
        output = {
            "masks": results.masks.data.cpu().numpy(),    # (N, H, W)
            "boxes": results.boxes.xyxy.cpu().numpy(),    # (N, 4)
            "confs": results.boxes.conf.cpu().numpy(),    # (N,)
            "cls_ids": results.boxes.cls.cpu().numpy().astype(int),
            "names": results.names
        }
        return output
        
    def display_frame(self, frame, output):
        overlay = frame.copy()
        
        if output['masks'] is None or len(output['boxes']) == 0:
            return frame
        
        h, w = frame.shape[:2]

        for mask, box, conf, cls_id in zip(output['masks'], output['boxes'], output['confs'], output['cls_ids']):
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # Resize & threshold mask
            mask = cv2.resize(mask, (w, h))
            binary_mask = mask > 0.5

            # Apply transparent color mask
            overlay[binary_mask] = (
                0.5 * overlay[binary_mask] + 0.5 * np.array(color)
            ).astype(np.uint8)

            # Bounding box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{output['names'][cls_id]} {conf:.2f}"
            cv2.putText(
                overlay,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        return blended


if __name__ == "__main__":
    video_path = "../Videos/demo.mp4"
    segmentor = YOLOSegmentationDisplay(
        model_path="./models/yolov8l-seg.pt",
        conf_threshold=0.5
    )
    count=0
    skip_frames=10
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % skip_frames != 0:
            continue
        segmented_results = segmentor.segment_frame(frame)
        segmented_frame_overlay = segmentor.display_frame(frame, segmented_results)

        cv2.imshow("YOLOv8 Segmentation", segmented_frame_overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

