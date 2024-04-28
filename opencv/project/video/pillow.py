# import time
# from PIL import ImageGrab
# count =0

# s = time.time()
# while True:
#     img = ImageGrab.grab()
#     count+=1
#     if count==100:
#         break
# e = time.time()

# l = e-s
# print(l)


import mss
import time
# Get the screen resolution
with mss.mss() as sct:
    monitor = sct.monitors[1]
    screen_width = monitor["width"]
    screen_height = monitor["height"]
s = time.time()
count = 0
# Capture frames
with mss.mss() as sct:
        while True:

            # Capture the entire screen
            img = sct.grab(sct.monitors[1])
            count+=1
            if count==100:
                break
e = time.time()

l = e-s
print(l)
