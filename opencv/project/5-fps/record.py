import cv2
import pyautogui
import numpy as np
import time

# Set the screen resolution
SCREEN_SIZE = pyautogui.size()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("screen_record.avi", fourcc, 5.0, (SCREEN_SIZE.width, SCREEN_SIZE.height))

# Variable for frame interval (in seconds) to achieve desired FPS
frame_interval = 1 / 5

print(frame_interval)
# Variables for FPS calculation
start_time = time.time()
num_frames = 0

while True:
    # Record start time of processing a frame
    frame_start_time = time.time()

    # Capture the screen
    img = pyautogui.screenshot()

    # Convert the screenshot to a numpy array
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Write the frame to the output video file
    out.write(frame)

    # Calculate time taken to process the frame


    # Increment frame counter
    num_frames += 1

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # If one second has elapsed, calculate and print FPS
    if elapsed_time >= 1:
        fps = num_frames / elapsed_time
        print("FPS:", fps)
        num_frames = 0
        start_time = time.time()
    frame_time = time.time() - frame_start_time
    # Calculate the delay required to achieve desired FPS
    delay = frame_interval - frame_time
    print(f"delay {delay}")

    # Introduce the delay
    time.sleep(delay)

















