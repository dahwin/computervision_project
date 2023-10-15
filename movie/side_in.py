'''dahyun+darwin= dahwin'''

# from moviepy.editor import VideoFileClip
# import cv2
# import numpy as np
#
# clip1 = VideoFileClip('likey.mp4').subclip(0, 6)
# clip2 = VideoFileClip('feel.mp4').subclip(0, 6)
#
# duration = 30  # in frames
#
# frames1 = [frame for frame in clip1.iter_frames()][:-duration]
# duration_frames = [frame for frame in clip1.iter_frames()][-duration:]
#
# frames2 = [frame for frame in clip2.iter_frames()]
#
# height, width, _ = frames1[0].shape
# transition_frames = []
#
# for i in range(duration):
#     # Calculate the translation distance based on the frame index
#     # For slide-in effect from right, clip1 moves to the left and clip2 comes from the right
#     tx = int((width / duration) * i)
#     tx_clip1 = -tx
#     tx_clip2 = width - tx
#
#     # Create a blank frame
#     blank_frame = np.zeros_like(frames1[0])
#
#     # Apply translation to clip1
#     frame_clip1 = cv2.warpAffine(duration_frames[i], np.float32([[1, 0, tx_clip1], [0, 1, 0]]), (width, height))
#
#     # Apply translation to clip2
#     frame_clip2 = cv2.warpAffine(frames2[i], np.float32([[1, 0, tx_clip2], [0, 1, 0]]), (width, height))
#
#     alpha = i / duration  # Linearly increasing alpha value
#
#     blend_frame = cv2.addWeighted(frame_clip1, 1 - alpha, frame_clip2, alpha, 0)
#
#     transition_frames.append(blend_frame)
#
# remaining_frames = frames2[duration:]
#
# final_frames = frames1 + transition_frames + remaining_frames
# fps = clip1.fps
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
#
# for frame in final_frames:
#     output.write(frame)
#
# output.release()


'''perfect'''
# from moviepy.editor import VideoFileClip, vfx
#
# import cv2
# import numpy as np
#
# clip1 = VideoFileClip('likey.mp4').subclip(0, 6)
# clip1.resize(height=1080)
# clip1.set_position(("center", "center"))
# clip2 = VideoFileClip('motion.mp4').subclip(0, 6)
# clip2.resize(height=1080)
# clip2.set_position(("center", "center"))
#
#
# duration = 30  # in frames
#
# frames1 = [frame for frame in clip1.iter_frames()][:-duration]
# duration_frames = [frame for frame in clip1.iter_frames()][-duration:]
#
# frames2 = [frame for frame in clip2.iter_frames()]
#
# height, width, _ = frames1[0].shape
# transition_frames = []
#
# for i in range(duration):
#     # Calculate the translation distance based on the frame index
#     # For slide-in effect from right, clip1 moves to the left and clip2 comes from the right
#     tx = int((width / duration) * i)
#     tx_clip1 = -tx
#     tx_clip2 = width - tx
#
#     # Create a blank frame
#     blank_frame = np.zeros_like(frames1[0])
#
#     # # Convert duration frame from RGB to BGR
#     # frame_clip1 = cv2.cvtColor(duration_frames[i], cv2.COLOR_RGB2BGR)
#
#     # Apply translation to clip1
#     frame_clip1 = cv2.warpAffine(duration_frames[i], np.float32([[1, 0, tx_clip1], [0, 1, 0]]), (width, height))
#
#     # Apply translation to clip2
#     frame_clip2 = cv2.warpAffine(frames2[i], np.float32([[1, 0, tx_clip2], [0, 1, 0]]), (width, height))
#
#     alpha = i / duration  # Linearly increasing alpha value
#
#     blend_frame = cv2.addWeighted(frame_clip1, 1 - alpha, frame_clip2, alpha, 0)
#
#     transition_frames.append(blend_frame)
#
# remaining_frames = frames2[duration:]
#
# final_frames = frames1 + transition_frames + remaining_frames
# fps = clip1.fps
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output = cv2.VideoWriter('different.mp4', fourcc, fps, (width, height))
#
# for frame in final_frames:
#     # Convert frame from RGB to BGR
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     output.write(frame)
#
# output.release()

# experiment
from moviepy.editor import VideoFileClip, vfx

import cv2
import numpy as np

clip1 = VideoFileClip('likey.mp4').subclip(0, 6)
clip1.resize(height=1080)
clip1.set_position(("center", "center"))
clip2 = VideoFileClip('motion.mp4').subclip(0, 6)
clip2.resize(height=1080)
clip2.set_position(("center", "center"))


duration = 30  # in frames

frames1 = [frame for frame in clip1.iter_frames()][:-duration]
duration_frames = [frame for frame in clip1.iter_frames()][-duration:]

frames2 = [frame for frame in clip2.iter_frames()]

height, width, _ = frames1[0].shape
transition_frames = []

for i in range(duration):
    # Calculate the translation distance based on the frame index
    # For slide-in effect from right, clip1 moves to the left and clip2 comes from the right
    tx = int((width / duration) * i)
    tx_clip1 = -tx
    tx_clip2 = width - tx

    # Create a blank frame
    blank_frame = np.zeros_like(frames1[0])

    # # Convert duration frame from RGB to BGR
    # frame_clip1 = cv2.cvtColor(duration_frames[i], cv2.COLOR_RGB2BGR)

    # Apply translation to clip1
    frame_clip1 = cv2.warpAffine(duration_frames[i], np.float32([[1, 0, tx_clip1], [0, 1, 0]]), (width, height))

    # Apply translation to clip2
    frame_clip2 = cv2.warpAffine(frames2[i], np.float32([[1, 0, tx_clip2], [0, 1, 0]]), (width, height))

    alpha = i / duration  # Linearly increasing alpha value

    blend_frame = cv2.addWeighted(frame_clip1, 1 - alpha, frame_clip2, alpha, 0)

    transition_frames.append(blend_frame)

remaining_frames = frames2[duration:]

final_frames =  transition_frames
fps = clip1.fps

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter('different.mp4', fourcc, fps, (width, height))

for frame in final_frames:
    # Convert frame from RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    output.write(frame)

output.release()




# x = dir(np)
#
# print('\n'.join(x))
# print(len(x))