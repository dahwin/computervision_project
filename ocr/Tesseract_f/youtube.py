import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"
frames = cv2.imread('screenshot.png')
# video = cv2.VideoCapture("C:\\Users\\Pc\\Desktop\\conputer_Vison\\ocr\\dahwin.mp4")
#
# # Setting width and height for video feed
# video.set(3, 640)
# video.set(4, 480)

# Allows continuous frames
while True:
    # Capture each frame from the video feed
    # extra, frames = video.read()

    data = pytesseract.image_to_data(frames)
    print(pytesseract.image_to_string(frames))
    for z, a in enumerate(data.splitlines()):
        # Counter
        if z != 0:
            # Converts 'data1' string into a list stored in 'a'
            a = a.split()
            # Checking if array contains a word
            if len(a) == 12:
                # Storing values in the right variables
                x, y = int(a[6]), int(a[7])
                w, h = int(a[8]), int(a[9])
                # Display bounding box of each word
                cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Display detected word under each bounding box
                cv2.putText(frames, a[11], (x - 15, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
    # Output the bounding box with the image
    cv2.imshow('Video output', frames)

    # Check if the 'q' key was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
# video.release()
cv2.destroyAllWindows()
