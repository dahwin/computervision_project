'''dahyun+darwin = dahwin'''
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
prev_dahwin = 0
prev_time = time.time()
last_blink_time = time.time()
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
cap = cv2.VideoCapture(0)
rectangle_size = (940, 1080)
dahwin = 0
counter_right = 0
stage_left = None
stage_right = None
# Open the two video files

cap1 = cv2.VideoCapture("C:\\Users\\Pc\\Desktop\\conputer_Vison\\mdiapipe\project\\upwork\\Upwork_07_Chest_Open.mp4")
cap2 = cv2.VideoCapture("G:\\download\\Video\\YouCut_20230310_103050016.mp4")
# cap2 = cv2.VideoCapture("G:\\download\\Video\\[MPD직캠] 트와이스 다현 직캠 4K 'Feel Special' (TWICE DAHYUN FanCam) - @MCOUNTDOWN_2019.9.26.mp4")
# cap2 = cv2.VideoCapture("C:\\Users\\Pc\\Desktop\\conputer_Vison\\mdiapipe\\project\\upwork\\TIME_OR_REPEATER.mp4")

dah = 0
dahy = 0
ratee = 0
def dahyun0(d):
    global dah
    dah = d

def dahyun1(d):
    global dahy
    dahy = d
    ratee = (dah - dahy)



def rate(dah, dahy):

    global ratee
    ratee = dah -dahy


accuracy = 'Loading'


# Define the size of the rectangle for each video
rectangle_size = (940, 1080)

# Create the window for the merged video
cv2.namedWindow("Merged Video", cv2.WINDOW_NORMAL)

start_time = time.time()
cv2.resizeWindow("Merged Video", (rectangle_size[0]*2, rectangle_size[1]))
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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


                # Calculate angle
                l_angle = calculate_angle(r_index, nose, l_index)

                # Visualize angle
                cv2.putText(image, str(l_angle),
                            tuple(np.multiply(nose, [image.shape[1], image.shape[0]]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                            )
                # print(l_angle)
                if l_angle > 150:
                    stage_left = "open"
                if l_angle < 75 and stage_left == "open":
                    stage_left = "close"
                    dahwin += 1
                    # print(f'counter_left  :{dahwin}')
                # timee =time.time()
                # print(dahwin)
                # final_time = timee-start_time
                # print(final_time)
                # rate = final_time/dahwin
                # print(rate)

                if int(time.time()) % 1 == 0:
                    dahyun0(d=dahwin)
                if int(time.time()) % 2 == 0:
                    dahyun1(d=dahwin)
                # if int(time.time()) % 2 == 0:
                rate(dah=dah, dahy=dahy)

                try:
                    if ratee >= 1:
                        accuracy = 'High Accuracy'
                    else:
                        accuracy = 'Low Accuracy'
                except:
                    pass



                # print(dah)
                # print(dahy)
                # print(f"value of ratee {ratee}")
                # print(accuracy)


            except:
                pass
                # setup status bax
            cv2.rectangle(image, (0, 0), (400, 100), (0, 255, 255), -1)

            # Rep data
            cv2.putText(image, 'Count OF Repetitions', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(image, str(accuracy),
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