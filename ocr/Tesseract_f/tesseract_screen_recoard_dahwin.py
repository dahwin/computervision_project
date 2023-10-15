'''dahyun+darwin = dahwin'''
import datetime
import time
from PIL import ImageGrab
import numpy as np
import cv2
from win32api import GetSystemMetrics
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"

def main():
    width = GetSystemMetrics(0)
    height = GetSystemMetrics(1)
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    print(time_stamp)
    file_name = f'{time_stamp}.mp4'
    fourcc = cv2.VideoWriter_fourcc('m' , 'p' , '4' , 'v')
    # captured_video = cv2.VideoWriter(file_name , fourcc , 20.0 , (width , height))
    data = 0

    def take_screenshot():
        # img = ImageGrab.grab(bbox=(0, 0, width, height))
        img = ImageGrab.grab()

        img_np = np.array(img)
        img_final = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f'{time_stamp}_screenshot_{int(time.time())}.jpg', img_final)
        nonlocal data
        data = pytesseract.image_to_string(img_final)

    while True:
        # img = ImageGrab.grab(bbox=(0, 0, width, height))
        img = ImageGrab.grab()

        img_np = np.array(img)
        img_final = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        cv2.imshow('Dawin', img_final)
        # captured_video.write(img_final)

        if int(time.time()) % 2 == 0:
            take_screenshot()
            print(data)

        if cv2.waitKey(10) == ord(("r")):
            break

if __name__ == "__main__":
    main()
