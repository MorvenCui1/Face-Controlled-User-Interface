import cv2
import mediapipe as mp
from timeit import default_timer as timer

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

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
    
        # Calculating distance between top and bottom of mouth for clicking motion
        top_mouth = landmarks[13]
        bottom_mouth = landmarks[15]
        top_mouth_x = int(top_mouth.x * frame_w)
        top_mouth_y = int(top_mouth.y * frame_h)
        bottom_mouth_x = int(bottom_mouth.x * frame_w)
        bottom_mouth_y = int(bottom_mouth.y * frame_h)

        cv2.circle(frame, (top_mouth_x,top_mouth_y), 3, (0, 255, 0))
        cv2.circle(frame, (bottom_mouth_x,bottom_mouth_y), 3, (0, 255, 0))

        #Mouth Gap
        mouth_gap = top_mouth_y - bottom_mouth_y
        #print(mouth_gap)

        # Calculating left eye blinking for bringing up and down keyboard
        left = [landmarks[145], landmarks[159]]
        right = [landmarks[386], landmarks[374]]
        eyeGap = left[0].y - left[1].y
        print(eyeGap)

        #print(time)
        
            
    cv2.imshow('Face Camera', frame)
    cv2.waitKey(1)