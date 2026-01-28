import cv2
import orbslam3
import time

class ORB_SLAM3_Wrapper():
    def __init__(self, vocab_path="./ORB-SLAM3-python/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt", settings_path="./webcam.yaml"):
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

    def get_all_methods(self):
        methods = [method_name for method_name in dir(self.slam) 
           if callable(getattr(self.slam, method_name))]

        for m in methods:
            print(m)

    def run_slam(self, frame, timestamp):
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

        
    


            
