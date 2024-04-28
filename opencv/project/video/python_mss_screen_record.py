# import datetime
# import numpy as np
# import cv2
# import mss
# import time
# import pyautogui
# from moviepy.editor import ImageSequenceClip

# screen_width, screen_height = pyautogui.size()
# time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
# file_name = f'{time_stamp}.mp4'

# # Get the screen resolution
# with mss.mss() as sct:
#     monitor = sct.monitors[1]
#     screen_width = monitor["width"]
#     screen_height = monitor["height"]

# start_time = datetime.datetime.now()
# frame_count = 0
# frames = []

# # Capture frames
# with mss.mss() as sct:
#     while (datetime.datetime.now() - start_time).total_seconds() < 30:

#         # Capture the entire screen
#         img = np.array(sct.grab(sct.monitors[1]))

#         # Append the frame
#         frames.append(img)

#         # Increment frame count
#         frame_count += 1

# # Calculate FPS
# fps = frame_count / (datetime.datetime.now() - start_time).total_seconds()
# print("Actual FPS:", fps)

# # Convert frames from RGB to BGR
# frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

# # Convert frames to a movie clip
# video_clip = ImageSequenceClip(frames_bgr, fps=fps)

# # Write the video file
# video_clip.write_videofile(file_name)

# print("Video file written:", file_name)












import os

import datetime
import numpy as np
import cv2
import mss
import time
import pyautogui
from moviepy.editor import ImageSequenceClip

screen_width, screen_height = pyautogui.size()
time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
file_name = f'{time_stamp}.mp4'
# Create a directory to save images if it doesn't exist
output_dir = "frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)




# Get the screen resolution
with mss.mss() as sct:
    monitor = sct.monitors[1]
    screen_width = monitor["width"]
    screen_height = monitor["height"]

start_time = datetime.datetime.now()
frame_count = 0
frames = []
extracted_F= 10
fps = 40
all_frames = []

# Capture frames
with mss.mss() as sct:
    while (datetime.datetime.now() - start_time).total_seconds() < 30:

        # Capture the entire screen
        img = np.array(sct.grab(sct.monitors[1]))

        # Increment frame count
        frame_count += 1
        print(frame_count)
        res = frame_count % fps
        print(res)
        d = fps/extracted_F
        frame_count+=1
        if res%d==0.0:
            print(True)
            all_frames.append(img)

        # cv2.imshow('frame',img_final)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        if frame_count==300:
            break



# Calculate FPS
fps = frame_count / (datetime.datetime.now() - start_time).total_seconds()
print("Actual FPS:", fps)

# # Convert frames from RGB to BGR
frames_bgr = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
# Save only the frames in all_frames list
for idx, frame in enumerate(frames_bgr):
    frame_path = os.path.join(output_dir, f"frame_{idx:05d}.png")
    cv2.imwrite(frame_path, frame)

print(len(all_frames))