import datetime
import numpy as np
import cv2
import pyautogui

screen_width, screen_height = pyautogui.size()
time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
print(time_stamp)
file_name = f'{time_stamp}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
captured_video = cv2.VideoWriter(file_name, fourcc, 25, (screen_width, screen_height))
import time
start_time = time.time()
frame_count = 0
while True:
    img = pyautogui.screenshot()
    img = np.array(img)
    img_final = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    captured_video.write(img_final)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    frame_count += 1
    if time.time() - start_time >= 30:
        break

# Calculate FPS
fps = frame_count / (time.time() - start_time)
print("Estimated FPS:", fps)

# Release the VideoWriter
captured_video.release()



# import datetime
# import numpy as np
# import cv2
# import pyautogui

# # Get the screen resolution
# screen_width, screen_height = pyautogui.size()


# start_time = datetime.datetime.now()
# frame_count = 0

# while (datetime.datetime.now() - start_time).total_seconds() < 30:
#     # Capture the entire screen
#     img = pyautogui.screenshot()
    
#     # Convert the screenshot to a numpy array
#     img = np.array(img)
    
#     # Convert RGB to BGR
#     img_final = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
#     # Increment frame count
#     frame_count += 1

# # Calculate FPS
# fps = frame_count / (datetime.datetime.now() - start_time).total_seconds()
# print("Estimated FPS:", fps)






