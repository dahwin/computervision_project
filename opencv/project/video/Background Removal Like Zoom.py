'''dahyun+darwin = dahwin'''
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
cap = cv2.VideoCapture('C:\\Users\\Pc\\Desktop\\conputer_Vison\\ocr\\dahwin.mp4')
cap.set(3,640)
cap.set(4,480)
segmentor = SelfiSegmentation()
while True:
    ret , img = cap.read()
    imgout = segmentor.removeBG(img,(0,0,0),threshold=0.98)
    imgstacked = cvzone.stackImages([img,imgout],2,1)
    cv2.imshow('dahwin',imgstacked)
    cv2.waitKey(1)