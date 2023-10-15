'''dahyun+darwin = dahwin'''
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize variables
last_nose_pos = None
last_r_index_pos = None
last_l_index_pos = None
last_left_foot_index_pos = None
last_right_foot_index_pos = None
last_time = time.time()
cap = cv2.VideoCapture(0)
# Function to calculate distance between two points
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

# Start video capture
rectangle_size = (940, 1080)

# Open the two video files

cap1 = cv2.VideoCapture("C:\\Users\\Pc\\Desktop\\conputer_Vison\\mdiapipe\project\\upwork\\Upwork_07_Chest_Open.mp4")
cap2 = cv2.VideoCapture("G:\\download\\Video\\YouCut_20230310_103050016.mp4")
# cap2 = cv2.VideoCapture("G:\\download\\Video\\[MPD직캠] 트와이스 다현 직캠 4K 'Feel Special' (TWICE DAHYUN FanCam) - @MCOUNTDOWN_2019.9.26.mp4")
# cap2 = cv2.VideoCapture("C:\\Users\\Pc\\Desktop\\conputer_Vison\\mdiapipe\\project\\upwork\\TIME_OR_REPEATER.mp4")



accuracy = 'Loading'


# Define the size of the rectangle for each video
rectangle_size = (940, 1080)

# Create the window for the merged video
cv2.namedWindow("Merged Video", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Merged Video", (rectangle_size[0]*2, rectangle_size[1]))
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap1.isOpened() and cap2.isOpened() :
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 and ret2:
            # Resize each frame to fit the rectangle
            frame1 = cv2.resize(frame1, rectangle_size)
            frame2 = cv2.resize(frame2, rectangle_size)
            # Recolor image to RGB
            image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if not results.pose_landmarks:  # Check if pose detection failed
                continue

            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract the coordinates of the nose, right index finger, and left index finger landmarks
            curr_nose_pos = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                             results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y]
            curr_r_index_pos = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y]
            curr_l_index_pos = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y]
            curr_left_foot_index_pos = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
            curr_right_foot_index_pos = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]

        if ret1 and ret2:
            # Resize each frame to fit the rectangle
            frame1 = cv2.resize(frame1, rectangle_size)
            frame2 = cv2.resize(frame2, rectangle_size)
            # Recolor image to RGB
            image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if not results.pose_landmarks:  # Check if pose detection failed
                continue

            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract the coordinates of the nose, right index finger, and left index finger landmarks
            curr_nose_pos = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                             results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y]
            curr_r_index_pos = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y]
            curr_l_index_pos = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y]
            curr_left_foot_index_pos = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
            curr_right_foot_index_pos = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y]

        # Calculate the distance traveled by the nose, right index finger, and left index finger
        if last_nose_pos is not None:
            nose_dist = distance(curr_nose_pos, last_nose_pos)
            r_index_dist = distance(curr_r_index_pos, last_r_index_pos)
            l_index_dist = distance(curr_l_index_pos, last_l_index_pos)
            left_foot_index_dist = distance(curr_left_foot_index_pos, last_left_foot_index_pos)
            right_foot_index_dist = distance(curr_right_foot_index_pos, last_right_foot_index_pos)

            # Calculate the time elapsed
            curr_time = time.time()
            time_elapsed = curr_time - last_time

            # Calculate the overall speed of the pose
            overall_speed = (nose_dist + r_index_dist + l_index_dist + right_foot_index_dist + left_foot_index_dist) / time_elapsed

            # Determine whether the pose is fast or slow based on the overall speed
            if overall_speed > 0.6:
                print('Fast')
            else:
                print('Slow')
            # Update the last positions and time
            last_nose_pos = curr_nose_pos
            last_r_index_pos = curr_r_index_pos
            last_l_index_pos = curr_l_index_pos
            last_left_foot_index_pos = curr_left_foot_index_pos
            last_right_foot_index_pos = curr_right_foot_index_pos
            last_time = time.time()
            print(accuracy)

                # setup status bax
            cv2.rectangle(image, (0, 0), (400, 100), (0, 255, 255), -1)

            # Rep data
            cv2.putText(image, 'Count OF Repetitions', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(image, str(curr_nose_pos),
                        (160, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            # Concatenate the two frames horizontally
            merged_frame = np.concatenate((frame1,image), axis=1)

            cv2.imshow("Merged Video", merged_frame)
        else:
            # Restart video playback if either video reaches the end
            if not ret1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if not ret2:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()