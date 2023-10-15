'''dahyun+darwin= dahwin'''
import cv2
import numpy as  np
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dahwin.mp4',fourcc,20.0,(640,480))


# Create an empty list to store the frames
frames = []

# Add some sample frames to the list
for i in range(10):
    # Create a random numpy array representing a frame
    frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    frames.append(frame)

for frame in frames:
    frame = np.array(frame,dtype=np.uint8)
    out.write(frame)
out.release()

# write the first video with array, with dahwin
# happy dahwin project