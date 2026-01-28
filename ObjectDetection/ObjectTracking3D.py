import numpy as np
import open3d as o3d
import copy
from scipy.spatial import KDTree

class TrackedObject:
    _id_counter = 0

    def __init__(self, label, center, pcd, bbox):
        self.id = TrackedObject._id_counter
        TrackedObject._id_counter += 1
        
        self.label = label
        self.center = np.array(center) # World Coordinates
        self.pcd = pcd
        self.bbox = bbox
        self.hit_count = 1  # How many times we've seen this object
        
        # Color the point cloud uniquely for visualization
        self.color = np.random.rand(3)
        self.pcd.paint_uniform_color(self.color)
        self.bbox.color = self.color # Match box color to points

    def update(self, new_center, new_pcd):
        """Merge new data into existing object"""
        # 1. Update Position (Running Average)
        self.center = (self.center * self.hit_count + new_center) / (self.hit_count + 1)
        self.hit_count += 1
        
        # 2. Merge Point Clouds
        # We add the new points, then downsample to keep it lightweight
        self.pcd += new_pcd
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.02) # 2cm voxel grid
        self.pcd.paint_uniform_color(self.color) # Re-apply color
        
        # 3. Update Bounding Box
        # Re-compute box around the merged cloud
        self.bbox = self.pcd.get_oriented_bounding_box()
        self.bbox.color = self.color

class Semantic3DTracker:
    def __init__(self, distance_threshold=0.5):
        self.map_objects = [] # List of TrackedObject
        self.distance_threshold = distance_threshold
        self.dirty = False

    def process_frame(self, spatial_objects, camera_pose_matrix):
        """
        Main loop: 
        1. Transform detections to World Space
        2. Match with existing map objects
        3. Update or Create new objects
        """
        if camera_pose_matrix is None:
            return self.map_objects

        # 1. Transform to World Frame
        world_detections = self._transform_to_world(spatial_objects, camera_pose_matrix)
        
        # 2. Match and Update
        for det in world_detections:
            matched_obj = self._find_match(det)
            
            if matched_obj:
                matched_obj.update(det['center'], det['pcd'])
            else:
                self._create_new_object(det)
                
        return self.map_objects

    def _create_new_object(self, det):
        new_obj = TrackedObject(det['label'], det['center'], det['pcd'], det['bbox_o3d'])
        self.map_objects.append(new_obj)
        self.dirty = True

    def _transform_to_world(self, spatial_objects, pose):
        """
        Converts Camera-Space centers/PCDs to World-Space using the pose matrix.
        Manual geometric transformation to avoid Open3D errors.
        """
        world_dets = []
        
        # Extract Rotation (3x3) and Translation (3) from Pose (4x4)
        R_pose = pose[:3, :3]
        T_pose = pose[:3, 3]

        for obj in spatial_objects:
            # A. Transform Center
            center_cam = np.array(obj['center'])
            # center_world = R * center + T
            center_world = (R_pose @ center_cam) + T_pose
            
            # B. Transform Point Cloud
            # We must use deepcopy because transform acts in-place
            pcd_world = copy.deepcopy(obj['pcd'])
            pcd_world.transform(pose)
            
            # C. Transform Bounding Box (The Fix)
            # Instead of obj['bbox'].transform(pose), we reconstruct it.
            bbox_old = obj['bbox_o3d']
            
            # New Rotation = Pose_Rotation * Old_Rotation
            new_R = R_pose @ bbox_old.R 
            
            # Create new box with World Center, World Rotation, and Original Extent
            bbox_world = o3d.geometry.OrientedBoundingBox(center_world, new_R, bbox_old.extent)
            bbox_world.color = bbox_old.color

            world_dets.append({
                'label': obj['label'],
                'center': center_world,
                'pcd': pcd_world,
                'bbox_o3d': bbox_world
            })
            
        return world_dets

    def _find_match(self, det):
        """Simple distance-based matching"""
        if not self.map_objects:
            return None
            
        # Filter by label first (only match 'chair' with 'chair')
        candidates = [obj for obj in self.map_objects if obj.label == det['label']]
        if not candidates:
            return None
            
        # Find closest candidate
        centers = np.array([obj.center for obj in candidates])
        
        # Use KDTree for efficient nearest neighbor (overkill for small lists but good practice)
        tree = KDTree(centers)
        dist, idx = tree.query(det['center'])
        
        if dist < self.distance_threshold:
            return candidates[idx]
            
        return None