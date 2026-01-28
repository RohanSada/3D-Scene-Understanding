import numpy as np
import open3d as o3d
import time
import queue
from ObjectTracking3D import Semantic3DTracker

def controller_process(ctrl_queue):
    tracker = Semantic3DTracker(distance_threshold=0.5)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Global Object Map", width=960, height=720)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 5.0
    
    # 1. Add a persistent Origin Frame (so you always see SOMETHING)
    #    Red=X, Green=Y, Blue=Z
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(origin)

    pose_buffer = {} 
    obj_buffer = {}

    first_data_rendered = False

    while True:
        try:
            msg = ctrl_queue.get(timeout=0.01)
            
            if msg == "STOP":
                break

            msg_type = msg.get("MsgType")
            content = msg.get("content")
            
            ts_key = str(content.get("frame_timestamp"))

            if msg_type == "SLAM_POSE":
                pose_buffer[ts_key] = content['pose']
            elif msg_type == "OBJECTS_3D":
                obj_buffer[ts_key] = content['Objects']
                #pose_buffer[ts_key] = np.eye(4) # Dummy Pose

            ready_timestamps = [ts for ts in pose_buffer if ts in obj_buffer]
            
            for ts in ready_timestamps:
                item1 = pose_buffer.pop(ts)
                item2 = obj_buffer.pop(ts)
                
                pose = None
                raw_objects = []

                if isinstance(item1, list) and len(item1) > 0 and isinstance(item1[0], dict):
                    raw_objects = item1
                    pose = item2
                elif isinstance(item2, list) and len(item2) > 0 and isinstance(item2[0], dict):
                    raw_objects = item2
                    pose = item1
                else:
                    if isinstance(item1, (np.ndarray, list)) and np.shape(item1) == (4,4):
                        pose = item1
                        raw_objects = item2 
                    else:
                        pose = item2
                        raw_objects = item1

                formatted_objects = []
                for obj in raw_objects:
                    if not isinstance(obj, dict): continue 

                    center = obj["center"] 
                    rotation = obj["rotation"]
                    extent = obj["dimensions"]
                    
                    bbox_o3d = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
                    bbox_o3d.color = obj.get("box_color", [0, 1, 0])
                    
                    pcd = o3d.geometry.PointCloud()
                    if "pcd_points" in obj and len(obj["pcd_points"]) > 0:
                        pcd.points = o3d.utility.Vector3dVector(obj["pcd_points"])
                        pcd.paint_uniform_color(bbox_o3d.color)
                    else:
                        pcd = bbox_o3d.sample_points_poisson_disk(number_of_points=50)
                        pcd.paint_uniform_color(bbox_o3d.color)

                    formatted_objects.append({
                        "label": obj["label"],
                        "center": center,
                        "bbox_o3d": bbox_o3d,
                        "pcd": pcd
                    })

                global_objects = tracker.process_frame(formatted_objects, camera_pose_matrix=pose)
                
                # --- VISUALIZATION UPDATE ---
                vis.clear_geometries()
                vis.add_geometry(origin, reset_bounding_box=False) # Always keep origin
                
                for track_obj in global_objects:
                    # Decide if we need to reset the camera
                    # We reset it ONLY on the first frame that has objects
                    reset_cam = False
                    if not first_data_rendered:
                        reset_cam = True
                        
                    vis.add_geometry(track_obj.pcd, reset_bounding_box=reset_cam)
                    vis.add_geometry(track_obj.bbox, reset_bounding_box=reset_cam)

                if len(global_objects) > 0:
                    first_data_rendered = True
                    print(f"Frame {ts}: Tracking {len(global_objects)} objects")

            # Cleanup
            if len(pose_buffer) > 50:
                del pose_buffer[min(pose_buffer.keys())]
            if len(obj_buffer) > 50:
                del obj_buffer[min(obj_buffer.keys())]

        except queue.Empty:
            pass
        
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()