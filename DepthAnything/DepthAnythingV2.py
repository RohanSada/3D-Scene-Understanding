import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthEstimator:
    """
    Depth estimation using Depth Anything v2
    """
    def __init__(self, model_size='small', device=None, metric=True):
        """
        Initialize the depth estimator
        
        Args:
            model_size (str): Model size ('small', 'base', 'large')
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
        """
        self.metric=metric
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Set MPS fallback for operations not supported on Apple Silicon
        if self.device == 'mps':
            print("Using MPS device with CPU fallback for unsupported operations")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            # For Depth Anything v2, we'll use CPU directly due to MPS compatibility issues
            self.pipe_device = 'cpu'
            print("Forcing CPU for depth estimation pipeline due to MPS compatibility issues")
        else:
            self.pipe_device = self.device
        
        print(f"Using device: {self.device} for depth estimation (pipeline on {self.pipe_device})")
        
        # Map model size to model name
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        self.load_metric_model(model_size)
        model_name = model_map.get(model_size.lower(), model_map['small'])
        
        # Create pipeline
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"Loaded Depth Anything v2 {model_size} model on {self.pipe_device}")
        except Exception as e:
            # Fallback to CPU if there are issues
            print(f"Error loading model on {self.pipe_device}: {e}")
            print("Falling back to CPU for depth estimation")
            self.pipe_device = 'cpu'
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"Loaded Depth Anything v2 {model_size} model on CPU (fallback)")
    
    def load_metric_model(self, model_size, domain='indoor'):
        checkpoints = {
            'indoor': {
                'small': "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                'base':  "depth-anything/Depth-Anything-V2-Metric-Hypersim-Base",
                'large': "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
            },
            'outdoor': {
                'small': "depth-anything/Depth-Anything-V2-Metric-VKITTI-Small",
                'base':  "depth-anything/Depth-Anything-V2-Metric-VKITTI-Base",
                'large': "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large",
            }
        }
        model_id = checkpoints[domain][model_size]
        print(f"Loading {model_id} on {self.device}...")

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def estimate_metric_depth(self, image_input):
        # Input Handling (CPU Side)
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Input must be a path or numpy array.")

        # Preprocessing
        # We manually handle the move to device to control types strictly
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # If using FP16 on CUDA, ensure inputs are also FP16 if the model expects it
        # (AutoImageProcessor usually outputs FP32, so we cast if needed)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Post-processing
        # Interpolate back to original size
        # 'bilinear' is faster than 'bicubic' and usually sufficient for depth maps
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1], 
            mode="bilinear", 
            align_corners=False,
        )

        # Return Numpy array (Move to CPU)
        return prediction.squeeze().float().cpu().numpy()

    def estimate_depth(self, image):
        if self.metric==True:
            depth_map = self.estimate_metric_depth(image)
        else:
            depth_map = self.estimate_normalized_depth(image)

        return depth_map
    
    def estimate_normalized_depth(self, image):
        """
        Estimate depth from an image
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            
        Returns:
            numpy.ndarray: Depth map (normalized to 0-1)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Get depth map
        try:
            depth_result = self.pipe(pil_image)
            depth_map = depth_result["depth"]
            
            # Convert PIL Image to numpy array if needed
            if isinstance(depth_map, Image.Image):
                depth_map = np.array(depth_map)
            elif isinstance(depth_map, torch.Tensor):
                depth_map = depth_map.cpu().numpy()
        except RuntimeError as e:
            # Handle potential MPS errors during inference
            if self.device == 'mps':
                print(f"MPS error during depth estimation: {e}")
                print("Temporarily falling back to CPU for this frame")
                # Create a CPU pipeline for this frame
                cpu_pipe = pipeline(task="depth-estimation", model=self.pipe.model.config._name_or_path, device='cpu')
                depth_result = cpu_pipe(pil_image)
                depth_map = depth_result["depth"]
                
                # Convert PIL Image to numpy array if needed
                if isinstance(depth_map, Image.Image):
                    depth_map = np.array(depth_map)
                elif isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
            else:
                # Re-raise the error if not MPS
                raise
        
        # Normalize depth map to 0-1
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map
    
    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        Colorize depth map for visualization
        
        Args:
            depth_map (numpy.ndarray): Depth map (normalized to 0-1)
            cmap (int): OpenCV colormap
            
        Returns:
            numpy.ndarray: Colorized depth map (BGR format)
        """
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_map_uint8, cmap)
        return colored_depth
    
    def get_depth_at_point(self, depth_map, x, y):
        """
        Get depth value at a specific point
        
        Args:
            depth_map (numpy.ndarray): Depth map
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            float: Depth value at (x, y)
        """
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        return 0.0
    
    def get_depth_in_region(self, depth_map, bbox, method='median'):
        """
        Get depth value in a region defined by a bounding box
        
        Args:
            depth_map (numpy.ndarray): Depth map
            bbox (list): Bounding box [x1, y1, x2, y2]
            method (str): Method to compute depth ('median', 'mean', 'min')
            
        Returns:
            float: Depth value in the region
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)
        
        # Extract region
        region = depth_map[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        # Compute depth based on method
        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        else:
            return float(np.median(region)) 

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from ObjectDetection.yolo_open_vocab import * 

    frame_path = '../Videos/Frames/frame_00320.jpg'

    try: 
        frame = cv2.imread(frame_path)
    except:
        print("Unable to load image from path")
    
    yolo_engine = YoloWorldMac()
    depth_engine = DepthEstimator(metric=True)

    depth_map = depth_engine.estimate_depth(frame)
    
    colourized_depth_map = depth_engine.colorize_depth(depth_map)

    yolo_results = yolo_engine.get_outputs(frame)

    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    yolo_cls = yolo_results[0].boxes.cls.cpu().numpy()
    yolo_conf = yolo_results[0].boxes.conf.cpu().numpy()
    label_names = yolo_results[0].names

    for bbox, cls_id in zip(yolo_boxes, yolo_cls):
        x1, y1, x2, y2 = map(int, bbox)
        obj_depth = depth_engine.get_depth_in_region(depth_map, bbox)
        cv2.rectangle(colourized_depth_map, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label_text = f"{label_names[int(cls_id)]} {obj_depth:.2f}m"
        cv2.putText(colourized_depth_map, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(label_text)
    cv2.imshow("col", colourized_depth_map)
    cv2.waitKey(0)
