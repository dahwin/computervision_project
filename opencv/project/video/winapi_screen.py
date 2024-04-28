import cv2
import os
import time
import numpy as np
from PIL import ImageGrab
from win32api import GetSystemMetrics

width = GetSystemMetrics(0)
height = GetSystemMetrics(1)

# Create a directory to save images if it doesn't exist
output_dir = "frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


s = time.time()
all_frames = []
frame_count = 0 
extracted_F= 5
fps = 10
while True:
    img = ImageGrab.grab()  # Capture the screen
    img = np.array(img)
    img_final = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(frame_count)
    res = frame_count % fps
    print(res)
    d = fps/extracted_F
    frame_count+=1
    if res%d==0.0:
        print(True)
        all_frames.append(img_final)

    # cv2.imshow('frame',img_final)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    if frame_count==300:
        break

e = time.time()
elapsed_time = e - s
fps = frame_count/elapsed_time
print("Elapsed Time:", elapsed_time)
print("Number of frames captured:", len(all_frames))
print(f'fps :{fps}')

# Save only the frames in all_frames list
for idx, frame in enumerate(all_frames):
    frame_path = os.path.join(output_dir, f"frame_{idx:05d}.png")
    cv2.imwrite(frame_path, frame)




# coun = 0
# for frame_index  in range(0,60):
#     fps=10
#     fps = round(fps)
#     res = frame_index % fps
#     print(frame_index)
#     print(res)
#     extracted_F= 10
#     d = fps/extracted_F
#     if res%d==0:  # Check if res is not zero before performing the modulo operation

#         print(True)
#         coun+=1
# print(f'total extracted frame {coun}')
