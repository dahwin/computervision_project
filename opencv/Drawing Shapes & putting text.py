import cv2
import numpy as np

blank = np.zeros((1920,1077,3), dtype='uint8')
blank[:] = 0,0,255

cv2.rectangle(blank,(0,0),(250,550),(0,250,0),thickness=2)
cv2.imshow('fucking windows', blank)

cv2.waitKey(0)