import numpy as np
import cv2 
import os
import time
import json

class CalibrationTool():
    def __init__(self, chessboardSize=(9,6), width=0.018):
        self.chessboardSize = chessboardSize
        self.width = width
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def calibrateFromImages(self, image_path, frame_size=(1920, 1080)):
        objp = np.zeros((self.chessboardSize[0] * self.chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chessboardSize[0],0:self.chessboardSize[1]].T.reshape(-1,2)
        objp = objp * self.width

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        img_arr = []
        for image in os.listdir(image_path):
            img = cv2.imread(os.path.join(image_path, image))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboardSize, None)

            if ret == True:
                img_arr.append(image)
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                imgpoints.append(corners)
                #cv2.drawChessboardCorners(img, self.chessboardSize, corners2, ret)
                #cv2.imshow('img', img)
                #cv2.waitKey(0)
            else:
                print("No Chessboard pattern found for img: ", image)

        cv2.destroyAllWindows()

        if len(objpoints) != 0:
            ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)
            calibError = self.validateCalibration(objpoints, imgpoints, cameraMatrix, dist, rvecs, tvecs, img_arr)
            print("Calibration Error: ", calibError)
            return ret, cameraMatrix, dist, rvecs, tvecs
        else:
            print("Calibration Failed.")
            return None, None, None, None, None

    def storeImages(self, img_path):
        cap = cv2.VideoCapture(0)
        num = 0
        while cap.isOpened():
            succes, img = cap.read()
            k = cv2.waitKey(5)
            if k == ord('b'):
                break
            elif k == ord('s'): 
                cv2.imwrite(img_path + 'img' + str(num) + '.png', img)
                print("image saved!")
                num += 1

            cv2.imshow('Img',img)

        cap.release()

        cv2.destroyAllWindows()

    def calibrateLive(self):
        count = 0
        cap = cv2.VideoCapture(0)        
        objp = np.zeros((self.chessboardSize[0] * self.chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chessboardSize[0],0:self.chessboardSize[1]].T.reshape(-1,2)
        objp = objp * self.width

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        last_processed_time = time.time()
        while True:
            ret, img = cap.read()
            if not ret:
                print("Error")
                break
            k = cv2.waitKey(5)
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            cur_time = time.time()
            frame_size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboardSize, None)
            if ret == True:
                if cur_time - last_processed_time > 1:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    last_processed_time = time.time()
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                cv2.drawChessboardCorners(img, self.chessboardSize, corners2, ret)
            cv2.imshow("Frame", img)

        if len(objpoints) != 0:
            ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)
            calibError = self.validateCalibration(objpoints, imgpoints, cameraMatrix, dist, rvecs, tvecs)
            print("Calibration Error auto: ", calibError)
            return ret, cameraMatrix, dist, rvecs, tvecs
        else:
            print("Calibration Failed.")
            return None, None, None, None, None

    def obtainTilt(self, rvecs):
        tilt_angles = []
        for i in rvecs: 
            rotation_matrix, _ = cv2.Rodrigues(i)
            pitch = np.arccos(rotation_matrix[2, 2])
            pitch = np.degrees(pitch)
            tilt_angles.append(pitch)
        return np.median(tilt_angles)

    def validateCalibration(self, objpoints, imgpoints, cameraMatrix, dist, rvecs, tvecs, img_arr=None):
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            if img_arr!=None:
                print("Error: ", error, "  For image: ", img_arr[i])
            mean_error += error
        return (mean_error/len(objpoints))

if __name__ == "__main__":
    calibration_path = './camera_calibration.json'
    cam_calib = CalibrationTool()
    ret, cameraMatrix, dist, rvecs, tvecs = cam_calib.calibrateFromImages('./Calibration_Images/', frame_size=(2160, 3840))
    #ret, cameraMatrix, dist, rvecs, tvecs = cam_calib.calibrateLive()
    data = {
            "camera_matrix": cameraMatrix.tolist(),
            "dist_coeffs": dist.tolist(),
            "rms_error": ret
        }
        
    with open(calibration_path, 'w') as f:
        json.dump(data, f, indent=4)