'''dahyun+darwin = dahwin'''
import cv2
import pytesseract
import time

start = time.time()
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ALL USER\Desktop\computervision_project\Tesseract-OCR\tesseract.exe"
text  = pytesseract.image_to_string('dahyun.jpg').lower()
# input = 'ashmawy1996'.lower()
# if input in text:
#     print(True)
# end = time.time()
# print(end-start)
print(text)

