'''dahyun+darwin = dahwin'''
import cv2
import pytesseract
import time

start = time.time()
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"
text  = pytesseract.image_to_string('fiverr1-.png').lower()
input = 'ashmawy1996'.lower()
if input in text:
    print(True)
end = time.time()
print(end-start)


