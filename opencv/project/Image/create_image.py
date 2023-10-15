'''dahyun+darwin = dahwin'''
# from PIL import Image
# import numpy as np
# data = np.random.rand(256,256,3)*255
# img = Image.fromarray(data.astype('uint8'),'RGB')
# img.save('dahwin.jpg')

import cv2
import numpy as np

# Create a numpy array
data = np.random.rand(256, 256, 3) * 255
data = np.clip(data, 0, 255).astype(np.uint8)
img = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)


# Save the image
cv2.imwrite("example.jpg", img)


# This is my first image created by array.
#I'm going to execute in the name of my soulmate dahyun
# happy dahwin project


