'''dahyun+darwin = dahwin'''
import cv2

# Read the image
img = cv2.imread('dahyun.jpg')
import numpy as np
np.set_printoptions(threshold=np.inf)
np.savetxt("output.txt", img.flatten(), fmt='%d')

print(img)
print(len(img))
