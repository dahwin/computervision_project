import pyautogui
import time

# Define the desired FPS
fps = 30

try:
    while True:
        # Get and print the mouse position
        mouse_position = pyautogui.position()
        print("Mouse position:", mouse_position)

        # Wait for the specified time to achieve the desired FPS
        time.sleep(1 / fps)
except KeyboardInterrupt:
    print("Program terminated by user.")
