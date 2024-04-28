
# import cv2

# cap = cv2.VideoCapture(r"C:\Users\ALL USER\Desktop\e\website\api\fancy.mp4")
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# total_length_seconds = frame_count / fps
# total_length_minutes = int(total_length_seconds // 60)
# total_length_seconds %= 60
# print("Height:", height, "Width:", width)
# print("FPS:", fps)
# print("Total Length: {} minutes and {:.2f} seconds".format(total_length_minutes, total_length_seconds))
# import time
# s = time.time()
# while True:
#     ret, frame = cap.read()
#     num = int(fps)

#     if not ret:
#         break

#     # Resize the frame to 1920x1080
#     # frame = cv2.resize(frame, (1280, 720))

#     cv2.imshow('frame', frame)

#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break
# e=time.time()
# l = e-s
# print(l)
# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import time

# cap = cv2.VideoCapture(r"C:\Users\ALL USER\Desktop\e\website\api\fancy.mp4")
# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# total_length_seconds = frame_count / fps
# total_length_minutes = int(total_length_seconds // 60)
# total_length_seconds %= 60
# print("Height:", height, "Width:", width)
# print("FPS:", fps)
# print("Total Length: {} minutes and {:.2f} seconds".format(total_length_minutes, total_length_seconds))

# s = time.time()
# all_frames = []

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
#     if frame_index % fps < 10:
#         all_frames.append(frame)

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break

# e = time.time()
# elapsed_time = e - s

# print("Elapsed Time:", elapsed_time)
# print("Number of frames captured:", len(all_frames))
# cap.release()
# cv2.destroyAllWindows()
import cv2
import os
import time

# Video file path
video_path = r"C:\Users\ALL USER\Desktop\e\website\api\fancy.mp4"

# Create a directory to save images if it doesn't exist
output_dir = "frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video capture
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_length_seconds = frame_count / fps
total_length_minutes = int(total_length_seconds // 60)
total_length_seconds %= 60
print("Height:", height, "Width:", width)
print("FPS:", fps)
print("Total Length: {} minutes and {:.2f} seconds".format(total_length_minutes, total_length_seconds))

s = time.time()
all_frames = []
extracted_F= 10
fps = round(fps)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(frame_index)
    res = frame_index % fps
    print(res)
    d = fps/extracted_F
    if res%d==0.0:
        print(True)
        all_frames.append(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

e = time.time()
elapsed_time = e - s

print("Elapsed Time:", elapsed_time)
print("Number of frames captured:", len(all_frames))

# Save only the frames in all_frames list
for idx, frame in enumerate(all_frames):
    frame_path = os.path.join(output_dir, f"frame_{idx:05d}.png")
    cv2.imwrite(frame_path, frame)

cap.release()
cv2.destroyAllWindows()
