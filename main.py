import cv2
import torch
import numpy as np
import open3d as o3d
import json
import time
import threading 
from SLAM.slam_wrapper import *
from ObjectDetection.ObjectTracking3D import *
from ObjectDetection.FuseSamYolo import *


class Controller():
    def __init__(self, vocab_path="./SLAM/ORB-SLAM3-python/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt", settings_path="./SLAM/webcam.yaml", camera_calibration_file='./utils/camera_calibration.json'):
        self.vocab_path = vocab_path
        self.settings_path = settings_path
        self.camera_calibration_file = camera_calibration_file

        self.slam_queue = queue.Queue()
        self.detection_queue = queue.Queue(maxsize=2)
        self.controller_queue = queue.Queue()

        self.slam = ORB_SLAM3_Wrapper(self.slam_queue, self.controller_queue, self.vocab_path, self.settings_path)
        self.object_detector = ObjectDetector3D(self.detection_queue, self.controller_queue, self.camera_calibration_file, stride=4)

        t_collect_results = threading.Thread(target=self.collect_results)
        t_collect_results.start()

        t_objdet = threading.Thread(target=self.object_detector.run)
        t_slam = threading.Thread(target=self.slam.run)
        t_slam.start()
        t_objdet.start()

    def collect_results(self):
        self.slam_buffer = {}
        self.detections_buffer = {}

        while True:
            try:
                message = self.controller_queue.get_nowait()
                print(message['MsgType'])
            except queue.Empty:
                continue

    def send_message(self, queue, message):
        try:
            queue.put(message)
        except queue.Error:
            print("Error sending message: ", message, " on queue:", queue)

    def get_frame_with_timestamp(self, cap):
        ret, frame = cap.read()
        return ret, frame, time.time()

    def create_and_send_slam_message(self, frame, frame_timestamp):
        message = {
            'MsgType': 'Input Frame',
            'content': {
                'frame': frame,
                'frame_timestamp': frame_timestamp
            }
        }
        self.send_message(self.slam_queue, message)

    def send_object_detection_message(self, frame, frame_timestamp):
        message = {
            'MsgType': 'Input Frame',
            'content': {
                'frame': frame,
                'frame_timestamp': frame_timestamp
            }
        }
        self.send_message(self.detection_queue, message)

    def start(self):
        start_time_flag = False
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while True: 
            if start_time_flag == False:
                self.slam.set_start_time()
                start_time_flag = True

            ret, frame, frame_timestamp = self.get_frame_with_timestamp(cap)
            if ret:
                self.create_and_send_slam_message(frame, frame_timestamp)
                self.send_object_detection_message(frame, frame_timestamp)

if __name__=='__main__':
    controller = Controller()
    controller.start()
