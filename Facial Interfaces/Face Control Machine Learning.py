import cv2
import mediapipe as mp

import pyautogui

from timeit import default_timer as timer

import os
import joblib

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

#Load machine learning models
base_dir = os.path.dirname(os.path.abspath(__file__))
eye_model_path = os.path.join(base_dir, "modelEye.pkl")
mouth_model_path = os.path.join(base_dir, "modelMouth.pkl")
eyeModel = joblib.load(eye_model_path)
mouthModel = joblib.load(mouth_model_path)

#Initialize flags for left eye
eye_close_left = False
startLeft = 0
endLeft = 0
#Initialize flags for right eye
eye_close_right = False
startRight = 0
endRight = 0
#Set flag for mouth
mouth_open = False
startOpen = 0
endOpen = 0

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    
    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Tracking nose for mouse movement
        pos = landmarks[1]
        nose_x = int(pos.x * frame_w)
        nose_y = int(pos.y * frame_h)
        cv2.circle(frame, (nose_x,nose_y), 3, (0, 255, 0))
        screen_x = screen_w / (frame_w) * nose_x
        screen_y = screen_h / (frame_h) * nose_y
        pyautogui.moveTo(screen_x, screen_y)
    
        #Mouth landmarks for keyboard
        top_mouth = landmarks[13]
        bottom_mouth = landmarks[15]
        top_mouth_x = int(top_mouth.x * frame_w)
        top_mouth_y = int(top_mouth.y * frame_h)
        bottom_mouth_x = int(bottom_mouth.x * frame_w)
        bottom_mouth_y = int(bottom_mouth.y * frame_h)

        cv2.circle(frame, (top_mouth_x,top_mouth_y), 3, (0, 255, 0))
        cv2.circle(frame, (bottom_mouth_x,bottom_mouth_y), 3, (0, 255, 0))

        mouth_gap = top_mouth_y - bottom_mouth_y
        if mouthModel.predict([[mouth_gap]]):
            if (mouth_open == False):
                startOpen = timer()
                mouth_open = True
        if not mouthModel.predict([[mouth_gap]]):
            if (mouth_open == True):
                endOpen = timer()
                mouth_open = False

        timeMouth = endOpen - startOpen
        if (timeMouth > 2):
            startOpen = timer()
            endOpen = timer()
            pyautogui.hotkey(['ctrl', 'win', 'o'])
            pyautogui.sleep(1)
        
        #Left eye landmarks for left click
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        eyeGapLeft = left[0].y - left[1].y
        if eyeModel.predict([[eyeGapLeft]]):
            if (eye_close_left == False):
                startLeft = timer()
                eye_close_left = True
        
        if not eyeModel.predict([[eyeGapLeft]]):
            if (eye_close_left == True):
                endLeft = timer()
                eye_close_left = False
        
        timeLeft = endLeft - startLeft
        if (timeLeft > 2):
            startLeft = timer()
            endLeft = timer()
            pyautogui.click()
            pyautogui.sleep(0.25)
        
        #Right eye landmarksfor right click
        right = [landmarks[374], landmarks[386]]
        eyeGapRight = right[0].y - right[1].y
        for landmark in right:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        eyeGapRight = right[0].y - right[1].y
        if eyeModel.predict([[eyeGapRight]]):
            if (eye_close_right == False):
                startRight = timer()
                eye_close_right = True
        
        if not eyeModel.predict([[eyeGapRight]]):
            if (eye_close_right == True):
                endRight = timer()
                eye_close_right = False
        
        timeRight = endRight - startRight
        if (timeRight > 2):
            startRight = timer()
            endRight = timer()
            pyautogui.rightClick()
            pyautogui.sleep(0.25)

    cv2.imshow('Face Camera', frame)
    cv2.waitKey(1)