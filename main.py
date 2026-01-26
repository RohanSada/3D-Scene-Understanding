import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import multiprocessing as mp
from processes.camera import camera_process
from processes.slam_process import slam_process
from processes.detection_process import detection_process
from processes.controller_process import controller_process

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    slam_queue = mp.Queue(maxsize=3)
    det_queue  = mp.Queue(maxsize=1)
    ctrl_queue = mp.Queue()

    vocab = "./SLAM/ORB-SLAM3-python/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    settings = "./SLAM/webcam.yaml"
    calib = "./utils/camera_calibration.json"

    p_cam = mp.Process(target=camera_process, args=(slam_queue, det_queue))
    p_slam = mp.Process(target=slam_process, args=(slam_queue, ctrl_queue, vocab, settings))
    #p_det = mp.Process(target=detection_process, args=(det_queue, ctrl_queue, calib))
    p_ctrl = mp.Process(target=controller_process, args=(ctrl_queue,))

    p_cam.start()
    p_slam.start()
    #p_det.start()
    p_ctrl.start()

    p_cam.join()
    p_slam.join()
    #p_det.join()
    p_ctrl.join()