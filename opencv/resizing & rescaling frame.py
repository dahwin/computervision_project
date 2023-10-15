import cv2
img = cv2.imread('media/icon.jpg')
def rescaleFrame(frame, scale=100):
    witdh = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)
    dimentions = (witdh, height)
    return  cv2.resize(frame, dimentions, interpolation=cv2.INTER_AREA)
resized = rescaleFrame(img)
cv2.imshow('dahyun', resized)
cv2.waitKey(1)

vid = cv2.VideoCapture("G:\\download\\Video\\TWICE _The Feels_ Choreography Video (Moving Ver.)_2.mkv")

while(True):
    ret, frame = vid.read()
    frame_resized = rescaleFrame(frame , scale=10)
    cv2.imshow('frame', frame)
    cv2.imshow('video resized', frame_resized)

    if cv2.waitKey(4) & 0xff == ord('g'):
        break
