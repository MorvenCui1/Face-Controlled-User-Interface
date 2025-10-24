import cv2
import mediapipe as mp
import pyautogui
from timeit import default_timer as timer

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

eye_close = False
start = 0
end = 0

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
    
        # Calculating distance between top and bottom of mouth for clicking motion
        top_mouth = landmarks[13]
        bottom_mouth = landmarks[15]
        top_mouth_x = int(top_mouth.x * frame_w)
        top_mouth_y = int(top_mouth.y * frame_h)
        bottom_mouth_x = int(bottom_mouth.x * frame_w)
        bottom_mouth_y = int(bottom_mouth.y * frame_h)

        cv2.circle(frame, (top_mouth_x,top_mouth_y), 3, (0, 255, 0))
        cv2.circle(frame, (bottom_mouth_x,bottom_mouth_y), 3, (0, 255, 0))

        mouth_gap = top_mouth_y - bottom_mouth_y
        if (mouth_gap <= -20):
            pyautogui.click()
            pyautogui.sleep(1)

        # Calculating left eye blinking for bringing up and down keyboard
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        if (left[0].y - left[1].y) < 0.017:
            if (eye_close == False):
                start = timer()
                eye_close = True
        
        if (left[0].y - left[1].y) > 0.017:
            if (eye_close == True):
                end = timer()
                eye_close = False
        
        time = end - start
        
        if (time >= 2):
            pyautogui.hotkey(['ctrl', 'win', 'o'])
            pyautogui.sleep(1)
        
    cv2.imshow('Face Camera', frame)
    cv2.waitKey(1)