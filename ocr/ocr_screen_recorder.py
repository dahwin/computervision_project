'''dahyun+darwin = dahwin'''
import cv2
import numpy as np
import win32api
import datetime
from PIL import ImageGrab
from paddleocr import PaddleOCR

ocr = PaddleOCR()
# Get screen dimensions
width = win32api.GetSystemMetrics(0)
height = win32api.GetSystemMetrics(1)
dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Define the fourCC code
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

# Create the video writer object
out = cv2.VideoWriter(f'{dt}.mp4', fourcc, 20.0, (width, height))
text_file = open(f"{dt}.txt", "w")
# Start recording
while True:
    # Capture the screen
    img = np.array(ImageGrab.grab(bbox=(0, 0, width, height)))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Write the frame to the video
    cv2.imshow('dahwin',gray)
    # Run text recognition on the frame
    results = ocr.ocr(gray)

    # Write the text to a file
    text_file.write("\n".join(results))
    text_file.write("\n")

    out.write(img)


    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and destroy all windows
out.release()
cv2.destroyAllWindows()
