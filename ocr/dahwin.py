'''dahyun+darwin = dahwin'''
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import cv2

ocr_moddle = PaddleOCR(lang='en')

img = 'dahwinorc.jpg'

result = ocr_moddle.ocr(img)
print(result)

