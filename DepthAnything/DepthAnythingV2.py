import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthAnythingV2:
    def __init__(self, model_size="small"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("[Depth] Device: NVIDIA GPU (CUDA)")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[Depth] Device: Mac GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            print("[Depth] Device: CPU (Slow)")

        model_ids = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base":  "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        
        checkpoint = model_ids.get(model_size, model_ids["small"])
        print(f"[Depth] Loading model: {checkpoint}...")
        
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(self.device)
        self.model.eval()
        print("[Depth] Model loaded successfully.")

    def get_depth(self, frame):
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1], # (Height, Width)
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        return prediction.cpu().numpy()

    def get_visual_map(self, raw_depth):
        depth_min = raw_depth.min()
        depth_max = raw_depth.max()
        
        if depth_max - depth_min > 0:
            depth_norm = (raw_depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(raw_depth)
        
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        return depth_colored

if __name__ == "__main__":
    depth_engine = DepthAnythingV2(model_size="small")
    
    video_path = "../Videos/demo.mp4" 
    
    # Try opening video, fallback to webcam if fails
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}, trying webcam...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No video source found.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Get Raw Depth
        raw_depth = depth_engine.get_depth(frame)
        
        # 2. Get Visualization
        vis_depth = depth_engine.get_visual_map(raw_depth)
        
        # 3. Show Side-by-Side
        # Resize depth to match frame if needed (should be same, but safety first)
        if vis_depth.shape[:2] != frame.shape[:2]:
            vis_depth = cv2.resize(vis_depth, (frame.shape[1], frame.shape[0]))
            
        combined = np.hstack((frame, vis_depth))
        cv2.imshow("RGB vs Depth (Fast)", combined)
        
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()