import sys
sys.path.append('./SLAM/')
sys.path.append('../')
from slam_wrapper import ORB_SLAM3_Wrapper
import time

def slam_process(slam_queue, ctrl_queue, vocab_path, settings_path, ready_event):
    slam = ORB_SLAM3_Wrapper(vocab_path, settings_path)
    ready_event.set()
    time.sleep(5)
    print("Starting SLAM process...")
    start_time_flag = False
    while True:
        message = slam_queue.get()
        frame, frame_timestamp = message['content']['frame'], message['content']['frame_timestamp']
        if start_time_flag == False:
            slam.set_start_time()
            start_time_flag = True
        pose = slam.run_slam(frame, frame_timestamp)
        if pose is not None:
            ctrl_queue.put({
                "MsgType": "SLAM_POSE",
                'content': {
                    'pose': pose,
                    'frame_timestamp':frame_timestamp
                }
            })