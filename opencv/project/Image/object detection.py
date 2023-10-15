import cv2
img = cv2.imread("C:\\Users\\Pc\\Desktop\\conputer_Vison\\opencv\\media\\selfie.jpg")
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
classids,confs,bbox = net.detect(img,confThreshold=0.5)
print(classids,bbox)


for classid, confidence ,box in zip(classids.flatten(),confs.flatten(),bbox):
    cv2.rectangle(img,box,(0,0,255),2)
    cv2.putText(img,classnames[classid-1],(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
cv2.imshow('d',img)

cv2.waitKey(0)