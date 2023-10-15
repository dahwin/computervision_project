import cv2

# Open the grayscale video
cap = cv2.VideoCapture("media\\1970.mp4")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_color_video.mp4", fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()

    color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Write the RGB frame to the output video
    out.write(color_frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break



# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
