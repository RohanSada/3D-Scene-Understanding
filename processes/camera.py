import cv2
import time

def create_and_send_message(frame, frame_timestamp, queue):
    message = {
        'MsgType': 'Camera_Frames',
        'content':{
            'frame': frame,
            'frame_timestamp':frame_timestamp
        }
    }
    queue.put_nowait(message)
    return

def camera_process(slam_queue, det_queue):
    cap = cv2.VideoCapture(0)
    cv2.setNumThreads(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_timestamp = time.time()

        if slam_queue.full():
            slam_queue.get_nowait()
        create_and_send_message(frame, frame_timestamp, slam_queue)

        if not det_queue.full():
            create_and_send_message(frame, frame_timestamp, det_queue)

