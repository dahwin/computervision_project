import cv2
import numpy
import time
import mss
from PIL import ImageGrab

def screen_record() -> int:
    # Get the screen resolution
    screen_width, screen_height = ImageGrab.grab().size

    mon = (0, 0, screen_width, screen_height)

    title = "[PIL.ImageGrab] FPS benchmark"
    fps = 0
    last_time = time.time()

    while time.time() - last_time < 1:
        img = numpy.asarray(ImageGrab.grab(bbox=mon))
        fps += 1

        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

    return fps

def screen_record_efficient() -> int:
    # Get the screen resolution
    screen_width, screen_height = mss.mss().monitors[0]["width"], mss.mss().monitors[0]["height"]

    mon = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}

    title = "[MSS] FPS benchmark"
    fps = 0
    sct = mss.mss()
    last_time = time.time()

    while time.time() - last_time < 1:
        img = numpy.asarray(sct.grab(mon))
        fps += 1

        cv2.imshow(title, img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

    return fps

print("PIL:", screen_record())
print("MSS:", screen_record_efficient())
