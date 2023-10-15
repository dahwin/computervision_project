import cv2

cap = cv2.VideoCapture("D:\\video\Video\Video\\twice\dahyun.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(height,width)
print(fps)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to 1920x1080
    # frame = cv2.resize(frame, (1920, 1080))

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

