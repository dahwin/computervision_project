'''dahyun+darwin = dahwin'''
import cv2
import time

from PIL import Image
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"

video_file = "C:\\Users\\Pc\\Desktop\\conputer_Vison\\ocr\\dahwin.mp4"
captured_video = cv2.VideoCapture(video_file)
# cap = cv2.VideoCapture(video_file)



def take_screenshot(frame):
    img_np = np.array(frame)
    img_final = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_final)
    print(pytesseract.image_to_string(img))



while captured_video.isOpened():
    ret, frame = captured_video.read()

    cv2.imshow('Video', frame)
    if int(time.time()) % 4 == 0:
        take_screenshot(frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break




captured_video.release()
cv2.destroyAllWindows()


