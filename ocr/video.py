import cv2
from paddleocr import PaddleOCR

ocr = PaddleOCR()

# Start capturing video frames
cap = cv2.VideoCapture('dahwin.mp4') # 0 for webcam, or specify a video file path

while True:
    # Capture a new video frame
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass the frame to PaddleOCR
    result = ocr.ocr(frame)

    # Display the recognized text on the frame
    cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("OCR", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera or close the video file
cap.release()
cv2.destroyAllWindows()
