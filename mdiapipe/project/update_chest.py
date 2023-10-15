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
nose = None
r_index = None
l_index = None
left_foot_index = None
right_foot_index = None
# Function to calculate distance between two points
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5



rectangle_size = (940, 1080)

# Open the two video files


# cap2 = cv2.VideoCapture("G:\\download\\Video\\[MPD직캠] 트와이스 다현 직캠 4K 'Feel Special' (TWICE DAHYUN FanCam) - @MCOUNTDOWN_2019.9.26.mp4")
# cap2 = cv2.VideoCapture("C:\\Users\\Pc\\Desktop\\conputer_Vison\\mdiapipe\\project\\upwork\\TIME_OR_REPEATER.mp4")



accuracy = 'Loading'



# Create the window for the merged video
cv2.namedWindow("Merged Video", cv2.WINDOW_NORMAL)

start_time = time.time()
cap1 = cv2.VideoCapture("chest.mp4")
cap2 = cv2.VideoCapture(0)
cv2.resizeWindow("Merged Video", (rectangle_size[0]*2, rectangle_size[1]))
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5) as pose:
    while(cap1.isOpened() and cap2.isOpened()):
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

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                r_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
                l_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
                left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x]



            except:
                pass
            if last_nose_pos is not None:
                nose_dist = distance(nose, last_nose_pos)
                r_index_dist = distance(r_index, last_r_index_pos)
                l_index_dist = distance(l_index, last_l_index_pos)
                left_foot_index_dist = distance(left_foot_index, last_left_foot_index_pos)
                right_foot_index_dist = distance(right_foot_index, last_right_foot_index_pos)

                # Calculate the time elapsed
                curr_time = time.time()
                time_elapsed = curr_time - last_time

                # Calculate the overall speed of the pose
                overall_speed = ( nose_dist + r_index_dist + l_index_dist +right_foot_index_dist+left_foot_index_dist) / time_elapsed

                # Determine whether the pose is fast or slow based on the overall speed
                if overall_speed > 0.3:
                    accuracy = 'High Accuracy'
                    print('very Fast')
                else:
                    accuracy = 'Low Accuracy'
                    print('Slow')
            # Update the last positions and time
            last_nose_pos = nose
            last_r_index_pos = r_index
            last_l_index_pos = l_index
            last_left_foot_index_pos = left_foot_index
            last_right_foot_index_pos = right_foot_index
            last_time = time.time()
            print(accuracy)
            # setup status bax
            cv2.rectangle(image, (0, 0), (400, 100), (0, 255, 255), -1)

            # Rep data
            cv2.putText(image, 'Your Accuracy Level', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(accuracy),
                        (125, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            # Concatenate the two frames horizontally
            merged_frame = np.concatenate((frame1,image), axis=1)

            cv2.imshow("Exercise accuracy tracking", merged_frame)
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