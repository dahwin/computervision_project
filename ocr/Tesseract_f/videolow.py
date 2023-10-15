import cv2
video_file = "C:\\Users\\Pc\\Desktop\\conputer_Vison\\ocr\\dahwin.mp4"
# Read the video
cap = cv2.VideoCapture(video_file)

# Get the frames per second (fps) of the input video
fps = cap.get(cv2.CAP_PROP_FPS)



# Loop over the frames
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # Resize the frame to 640x260
        frame = cv2.resize(frame, (1920, 1080))

        # Display the frame
        cv2.imshow("Resized Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the resources
cap.release()

cv2.destroyAllWindows()
