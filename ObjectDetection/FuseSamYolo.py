import sys
sys.path.append('../')
sys.path.append('./ObjectDetection/')
import cv2
import os
from runFastSam import FastSam
from yolo_open_vocab import YoloWorldMac
from DepthAnything.DepthAnythingV2 import *
import numpy as np
import time
import open3d as o3d
import json
import queue

class ObjectDetector3D:
    def __init__(self, input_queue, output_queue, calibration_file='./utils/camera_calibration.json', stride=4):
        """
        stride: Downsampling factor. 
                stride=2 is high quality (slower). 
                stride=4 is fast (good for real-time video).
        """
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available(): self.device = "mps"
        
        self.stride = stride
        
        self.K_raw, self.dist_coeffs = self.load_calibration(calibration_file)
        
        print("Loading Depth and Object Detection Models...")
        self.depth_engine = DepthEstimator()
        self.object_detector = FuseSamYolo()
        
        self.map1, self.map2 = None, None
        self.ray_tensor = None
        self.initialized = False

    def load_calibration(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return np.array(data["camera_matrix"], dtype=np.float32), \
               np.array(data["dist_coeffs"], dtype=np.float32)

    def init_geometry(self, h, w):
        """
        Runs once on the first frame. 
        Pre-calculates undistortion maps and 3D ray vectors.
        """
        # A. Optimal New Camera Matrix
        self.K_new, roi = cv2.getOptimalNewCameraMatrix(
            self.K_raw, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        # B. Undistortion Look-Up Tables
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K_raw, self.dist_coeffs, None, self.K_new, (w, h), cv2.CV_16SC2
        )

        # C. Ray Casting Grid
        # Adjust intrinsics for the stride
        fx = self.K_new[0, 0] / self.stride
        fy = self.K_new[1, 1] / self.stride
        cx = self.K_new[0, 2] / self.stride
        cy = self.K_new[1, 2] / self.stride
        
        h_s, w_s = h // self.stride, w // self.stride

        # Create Grid (Y, X)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(0, h_s, device=self.device),
            torch.arange(0, w_s, device=self.device),
            indexing='ij'
        )
        
        # Calculate normalized ray vectors: (pixel - center) / focal
        x_ray = (x_grid - cx) / fx
        y_ray = (y_grid - cy) / fy
        
        # Stack into (H, W, 3) tensor -> [x_ray, y_ray, 1.0]
        ones = torch.ones_like(x_ray)
        self.ray_tensor = torch.stack((x_ray, y_ray, ones), dim=-1)
        
        self.initialized = True

    def get_message(self):
        while True:
            try:
                message = self.input_queue.get(timeout=0.1)
                print("Message received by Object Detection Module: ", message)
                return message['content']
            except queue.Empty:
                continue

    def send_message(self, queue, message):
        try:
            queue.put(message)
        except queue.Error:
            print("Error sending message: ", message, " on queue:", queue)

    def create_and_send_detections_message(self, detections, frame_timestamp):
        message = {
            'MsgType': 'Object_Detections',
            'content':{
                'detections': detections, 
                'frame_timestamp': frame_timestamp
            }
        }
        self.send_message(self.output_queue, message)

    def run(self):
        while True:
            message = self.get_message()
            frame, frame_timestamp = message['frame'], message['frame_timestamp']
            detections = self.process_frame(frame)
            self.create_and_send_detections_message(detections, frame_timestamp)

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        if not self.initialized:
            self.init_geometry(h, w)

        frame_undistorted = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

        detections = self.object_detector.process_frame(frame_undistorted)
        if not detections:
            return [], frame_undistorted

        depth_raw = self.depth_engine.estimate_metric_depth(frame_undistorted)
        
        if isinstance(depth_raw, np.ndarray):
            depth_tensor = torch.from_numpy(depth_raw).to(self.device)
        else:
            depth_tensor = depth_raw.to(self.device)

        depth_strided = depth_tensor[::self.stride, ::self.stride]
        
        points_3d_grid = self.ray_tensor * depth_strided.unsqueeze(-1)
        
        points_flat = points_3d_grid.reshape(-1, 3)
        
        spatial_objects = []
        
        for det in detections:
            mask = det['mask']
            target_h, target_w = depth_strided.shape
            
            mask_s = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            
            mask_tensor = torch.from_numpy(mask_s).to(self.device).bool().flatten()
            
            valid_indices = mask_tensor & (points_flat[:, 2] > 0.1)
            
            if valid_indices.sum() < 50: 
                continue

            obj_points_cpu = points_flat[valid_indices].cpu().numpy()
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points_cpu)
            
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            if len(pcd.points) < 10: continue

            obb = pcd.get_oriented_bounding_box()
            obb.color = (1, 0, 0)
            
            center = obb.get_center()
            extent = obb.extent
            
            spatial_objects.append({
                "label": det.get('label', 'object'),
                "center": center,   # [x, y, z] location in camera space
                "dimensions": extent,
                "bbox_o3d": obb,    # Open3D Geometry object
                "pcd": pcd          # Open3D PointCloud object
            })
            
        return spatial_objects, frame_undistorted

class FuseSamYolo():
    def __init__(self, yolo_model='./models/yolov8s-world.pt', sam_model='./models/FastSAM-s.pt'):
        self.yolo = YoloWorldMac(model_path=yolo_model)
        self.sam = FastSam(model_path=sam_model, conf_threshold=0.6, iou_threshold=0.9)
        self.erode_kernel = np.ones((5, 5), np.uint8) 

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

    def process_frame(self, frame, erosion_iterations=5):
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
                    raw_mask = sam_masks[best_mask_idx]
                    
                    mask_uint8 = raw_mask.astype(np.uint8)
                    
                    if erosion_iterations > 0:
                        processed_mask = cv2.erode(mask_uint8, self.erode_kernel, iterations=erosion_iterations)
                    else:
                        processed_mask = mask_uint8

                    result_entry = {
                        "label": label_name,
                        "confidence": float(conf),
                        "box": y_box.tolist(),
                        "mask_box": sam_boxes[best_mask_idx].tolist(),
                        "mask": processed_mask,
                        "iou": float(best_iou)
                    }
                    fused_results.append(result_entry)  
        return fused_results
    
    def display_fused_results(self, frame, fused_results):
        annotated_frame = frame.copy()
        
        MASK_COLOR = (0, 255, 0)
        BOX_COLOR = (0, 255, 0)
        TEXT_COLOR = (255, 0, 0)
        
        for result in fused_results:
            label = result['label']
            conf = result['confidence']
            box = result['box']
            mask = result['mask']
            
            # Resize mask if needed (Erosion doesn't change W/H, but SAM might output smaller masks)
            if mask.shape[:2] != annotated_frame.shape[:2]:
                mask = cv2.resize(mask, (annotated_frame.shape[1], annotated_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            mask_bool = mask.astype(bool)
            
            # Visualization Logic
            color_layer = np.zeros_like(annotated_frame, dtype=np.uint8)
            color_layer[mask_bool] = MASK_COLOR
            alpha = 0.4
            
            roi = annotated_frame[mask_bool]
            if roi.size > 0: # Check to ensure mask isn't empty after erosion
                blended_roi = (roi * (1 - alpha) + np.array(MASK_COLOR) * alpha).astype(np.uint8)
                annotated_frame[mask_bool] = blended_roi

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            
            label_text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), BOX_COLOR, -1)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
                        
        return annotated_frame

if __name__=='__main__':
    video_path = '../Videos/demo.mp4'
    processor = SpatialMapProcessor(calibration_file='../utils/camera_calibration.json', stride=4)
    
    # Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600, window_name='3D Result')
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0 
    objects = []         
    clean_frame = None   

    print("Controls:")
    print("  SPACE - Continue to next processed frame (Make sure VIDEO window is focused!)")
    print("  ESC   - Quit program")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            clean_frame = frame 

            did_process = False

            # Run Processor every 60th frame
            if frame_count % 60 == 0:
                t0 = time.time()
                objects, clean_frame = processor.process_frame(frame)
                did_process = True 
                
                print(f"--- Processed Frame {frame_count} ---")
                for obj in objects:
                    c = obj['center']
                    print(f" - {obj['label']} at X:{c[0]:.2f} Y:{c[1]:.2f} Z:{c[2]:.2f}")

            # Update Visualization
            vis.clear_geometries()
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
            
            for obj in objects:
                vis.add_geometry(obj['bbox_o3d'])
                vis.add_geometry(obj['pcd']) 
                
            vis.poll_events()
            vis.update_renderer()
            
            # Show OpenCV Window
            cv2.imshow("Input", clean_frame)

            # --- CONTROL LOGIC ---
            
            if did_process:
                print(f">> PAUSED at frame {frame_count}. Click the 'Input' window and press SPACE.")
                while True:
                    vis.poll_events()
                    vis.update_renderer()
                    
                    # Wait 100ms for key (easier to catch than 1ms)
                    key = cv2.waitKey(100) & 0xFF
                    
                    # If you are pressing keys and nothing happens, uncomment this line to debug:
                    # if key != 255: print(f"Debug: Key pressed: {key}") 

                    if key == 32: # SPACE
                        print("Resuming...")
                        break
                    elif key == 27: # ESC
                        raise KeyboardInterrupt 
            else:
                # Fast forward normal frames
                if cv2.waitKey(1) & 0xFF == 27: # ESC
                    break
            
    except KeyboardInterrupt:
        print("\nQuitting...")
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        vis.destroy_window()