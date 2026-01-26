import cv2
import orbslam3
import time
import inspect
import queue

class ORB_SLAM3_Wrapper():
    def __init__(self, input_queue, output_queue, vocab_path="./ORB-SLAM3-python/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt", settings_path="./ORB-SLAM3-python/third_party/ORB_SLAM3/Examples/Monocular/EuRoC.yaml"):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.vocab_path = vocab_path
        self.settings_path = settings_path
        self.slam = orbslam3.system(vocab_path, settings_path, orbslam3.Sensor.MONOCULAR)
        self.slam.set_use_viewer(False)
        self.slam.initialize()
        self.start_time = None

    def set_start_time(self):
        if self.start_time == None:
            self.start_time = time.time()   
        else:
            print("Start time already set before")
        return

    def get_all_methods(self):
        methods = [method_name for method_name in dir(self.slam) 
           if callable(getattr(self.slam, method_name))]

        for m in methods:
            print(m)

    def get_message(self):
        while True:
            try:
                message = self.input_queue.get(timeout=0.1)
                print("Message received by SLAM Module: ", message)
                return message['content']
            except queue.Empty:
                continue

    def send_message(self, message):
        try:
            self.output_queue.put(message)
        except:
            print("Error sending message from SLAM module")
        return

    def run(self):
        while True:
            message = self.get_message()
            frame_timestamp = message['frame_timestamp']
            t_frame = float(frame_timestamp - self.start_time)
            tracking = self.slam.process_image_mono(message['frame'], t_frame)
            if tracking:
                pose = self.slam.get_pose()
            else:
                pose = []
            out_message = {
                'MsgType': 'SLAM_Pose',
                'content':{
                    'pose': pose,
                    'frame_timestamp': frame_timestamp
                }
            }
            self.send_message(out_message)

    def run_slam_demo(self, frame, timestamp):
        t_frame = float(timestamp - self.start_time)
        tracking = self.slam.process_image_mono(frame, t_frame)
        if tracking:
            pose = self.slam.get_pose()
            return pose
        return None

if __name__=='__main__':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    slam = ORB_SLAM3_Wrapper()
    slam.set_start_time()
    while True: 
        ret, frame = cap.read()
        pose = slam.run_slam(frame, time.time())
        print(pose, type(pose))

        
    


            
