'''dahyun+darwin= dahwin'''
import cv2
import numpy as np
import random
img = cv2.imread('dahyun.jpg')
#random pixel
# for i in range(100):
#     for j in range(img.shape[1]):
#         img[i][j] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
copyy = img[500]
cv2.imshow('dahyun',img)
cv2.waitKey(0)
cv2.destroyAllWindows()






#read singal row:
# np.set_printoptions(threshold=np.inf)
# print(img[0])
#
# print(len(img[0]))
# print(img.shape)