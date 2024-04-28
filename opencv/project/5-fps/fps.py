import os
import cv2
import pyautogui
import numpy as np
import time

# Create directory if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

# Set the screen resolution
SCREEN_SIZE = pyautogui.size()

# Variable for frame interval (in seconds) to achieve desired FPS
frame_interval = 1 / 5

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

    # Save the frame
    cv2.imwrite(os.path.join("images", f"frame_{num_frames}.png"), frame)

    # Calculate time taken to process the frame

    # Increment frame counter
    num_frames += 1

    if num_frames >= 5:
        break

# Calculate FPS
end_time = time.time()
fps = num_frames / (end_time - start_time)
print("FPS:", fps)
