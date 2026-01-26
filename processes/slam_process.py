import sys
sys.path.append('./SLAM/')
sys.path.append('../')
from slam_wrapper import ORB_SLAM3_Wrapper

def slam_process(slam_queue, ctrl_queue, vocab_path, settings_path):
    slam = ORB_SLAM3_Wrapper(vocab_path, settings_path)
    while True:
        message = slam_queue.get()
        frame, frame_timestamp = message['content']['frame'], message['content']['frame_timestamp']
        pose = slam.process_frame(frame, frame_timestamp)
        if pose is not None:
            ctrl_queue.put({
                "MsgType": "SLAM_POSE",
                'content': {
                    'pose': pose,
                    'frame_timestamp':frame_timestamp
                }
            })