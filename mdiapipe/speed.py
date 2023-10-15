import mediapipe as mp
import cv2
import time
accuracy = 'Loading'
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize variables
last_nose_pos = None
last_r_index_pos = None
last_l_index_pos = None
last_left_foot_index_pos = None
last_right_foot_index_pos = None
last_time = time.time()

# Function to calculate distance between two points
def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

# Start video capture
# cap = cv2.VideoCapture("G:\\download\\Video\\[MPD직캠] 트와이스 다현 직캠 4K 'Feel Special' (TWICE DAHYUN FanCam) - @MCOUNTDOWN_2019.9.26.mp4")
# cap = cv2.VideoCapture("G:\\download\\Video\\[안방1열 직캠4K_고음질] 트와이스 다현 '필스페셜' (TWICE DAHYUN 'Feel Special' Fancam)ㅣ@SBS Inkigayo_2019.9.29.mp4")
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a more intuitive selfie-view display
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find pose landmarks
        results = pose.process(image)

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
            overall_speed = (nose_dist + r_index_dist + l_index_dist+right_foot_index_dist+left_foot_index_dist) / time_elapsed

            # Determine whether the pose is fast or slow based on the overall speed
            if overall_speed > 0.9:
                accuracy = 'Fast'
                print('very Fast')
            else:
                accuracy = 'Low'
                print('Slow')
        # Update the last positions and time
        last_nose_pos = curr_nose_pos
        last_r_index_pos = curr_r_index_pos
        last_l_index_pos = curr_l_index_pos
        last_left_foot_index_pos = curr_left_foot_index_pos
        last_right_foot_index_pos = curr_right_foot_index_pos
        last_time = time.time()
        print(accuracy)


        # Display the image
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
