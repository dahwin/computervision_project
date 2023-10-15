'''dahyun+darwin= dahwin'''
import cv2
from moviepy.editor import VideoFileClip, vfx

# Load the video file
video_file = r"D:\video\Video\Video\twice\twice.mp4"
clip = VideoFileClip(video_file)

# Apply a video effect
clip_with_effect = clip.fx(vfx.rotate, angle=45)  # Rotate the video by 45 degrees

# Convert the video frames to OpenCV format
frame_generator = clip_with_effect.iter_frames()

# Display the video frames using OpenCV
while True:
    try:
        frame = next(frame_generator)
        cv2.imshow("Video", frame)

        # Check if the user wants to quit
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    except StopIteration:
        break

# Release the resources
cv2.destroyAllWindows()
clip.close()
clip_with_effect.close()
