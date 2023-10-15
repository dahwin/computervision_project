'''dahyun+darwin= dahwin'''
import cv2

# Open the video file
video = cv2.VideoCapture("original_video.mp4")

# Get the video properties
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("modified_video.mp4", fourcc, fps, (width, height))

# Loop through each frame of the video
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Perform some manipulation, for example, resize the video
    frame = cv2.resize(frame, (width//2, height//2))

    # Write the frame to the output video
    out.write(frame)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects
video.release()
out.release()
