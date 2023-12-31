
'''dahyun+darwin = dahwin'''

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
cap = cv2.VideoCapture("G:\\download\\Video\\[MPD직캠] 트와이스 다현 직캠 4K 'Feel Special' (TWICE DAHYUN FanCam) - @MCOUNTDOWN_2019.9.26.mp4")

counter_left = 0
counter_right = 0
stage_left = None
stage_right = None

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get left hand coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Get right hand coordinates
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate left hand angle
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Calculate right hand angle
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Visualize left hand angle
            # cv2.putText(image, f'Left hand angle: {left_angle}',
            #             (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if left_angle > 160:
                stage_left = "down"
            if left_angle < 30 and stage_left == "down":
                stage_left = "up"
                counter_left += 1
                print(f'Left hand count: {counter_left}')

            # Visualize right hand angle
            # cv2.putText(image, f'Right hand angle: {right_angle}',
            #             (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if right_angle > 160:
                stage_right = "down"
            if right_angle < 30 and stage_right == "down":
                stage_right = "up"
                counter_right += 1
                print(f'Right hand count: {counter_right}')

        except:
            pass

        # # setup status bax
        # cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        #
        # # Rep data
        # cv2.putText(image, 'REPS', (15, 12),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(counter_left),
        #             (10, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #
        # # Stage data
        # cv2.putText(image, 'STAGE', (65, 12),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, stage_left,
        #             (60, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #
        # # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()