import cv2

cap = cv2.VideoCapture("D:\\video\\for editing\\adobe pr & affect\\no copyright\\video (3).mp4")
# cap = cv2.VideoCapture("D:\\video\Video\Video\\twice\dahyun.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(height,width)
print(fps)
classnames = []
classfile = "C:\\Users\\Pc\\Desktop\\conputer_Vison\\opencv\\model\\mobilenetssd\\coco.names"
with open(classfile,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n' )

configpath = "C:\\Users\\Pc\\Desktop\\conputer_Vison\\opencv\\model\\mobilenetssd\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightspath = "C:\\Users\\Pc\\Desktop\\conputer_Vison\\opencv\\model\\mobilenetssd\\frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightspath,configpath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)

while True:
    ret, img = cap.read()

    if not ret:
        break

    # Resize the frame to 1920x1080
    img = cv2.resize(img, (1920,1080))
    classids, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classids, bbox)
    if len(classids) != 0:

        for classid, confidence, box in zip(classids.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, (0, 0, 255), 2)
            if classid > 0 and classid - 1 < len(classnames):
                cv2.putText(img, classnames[classid - 1], (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
