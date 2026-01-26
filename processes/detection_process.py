import sys
sys.path.append('./ObjectDetection/')
from FuseSamYolo import ObjectDetector3D
import torch

def detection_process(det_queue, ctrl_queue, calib_path):
    torch.set_grad_enabled(False)
    detector = ObjectDetector3D(calib_path)

    while True:
        try:
            message = det_queue.get(timeout=0.01)
            frame, frame_timestamp = message['content']['frame'], message['content']
        except:
            continue

        result = detector.process_frame(frame)
        ctrl_queue.put({
            "MsgType": "OBJECTS_3D",
            "content": {
                'Objects': result,
                'frame_timestamp': frame_timestamp
            }
        })