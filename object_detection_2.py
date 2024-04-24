import numpy as np 
import cv2
from imutils.video import VideoStream
import datetime
import argparse
import time
import imutils

prototxt_path = "MobileNetSSD_deploy_prototxt.txt" 
model_path = "MobileNetSSD_deploy.caffemodel" 
image_path = "car.jpg" 

conf_limit = 0.25 

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
"horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", 
"tv/monitor"] 

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...") 
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path) 

image = cv2.imread(image_path) 
(h, w) = image.shape[:2] 
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

print("Sending image through the network...") 
net.setInput(blob) 
detections = net.forward() 


for i in np.arange(0, detections.shape[2]): 
      # extract the confidence  
      confidence = detections[0, 0, i, 2] 

      if confidence > conf_limit: 
           idx = int(detections[0, 0, i, 1]) 
           box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) 
           (startX, startY, endX, endY) = box.astype("int")
           label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100) 
           print("{}".format(label))

           cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2) 
           y = startY - 15 if startY - 15 > 15 else startY + 15 
           cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)









video_path = None
buffsize = 64
indx = 0



# lower and upper boundaries of a "green" object in HSV color space
# I didn't chnage the variable name but i modified the color range
green_range = [(160, 50, 50), (180, 255, 255)]
# initialize the list of tracked points
path = np.zeros((buffsize, 2), dtype='int')

video_path = None
if video_path is None:
    vs = VideoStream().start()
# warm up the camera
    time.sleep(2)
else:
    vs = cv2.VideoCapture(video_path)

while True:
    frame = vs.read()
    frame = frame if video_path is None else frame[1]
    # if there is no more frame in the video break the loop
    if frame is None:
        break
    frame = imutils.resize(frame, width=500)

    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    image = cv2.imread(video_path) 
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, 
    (300, 300), 127.5)
    net.setInput(blob) 
    detections = net.forward()

#     mask = cv2.inRange(hsv, green_range[0], green_range[1])
#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)


    mask = cv2.image
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        # find the largest contour in the mask
        cnt = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        M = cv2.moments(cnt)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # update the path list
            if indx < buffsize:
                path[indx] = (center[0], center[1])
                indx += 1
            else:
                path[0:indx-1] = path[1:indx]
                path[indx-1] = (center[0], center[1])

        for i in range(1, len(path)):
            # otherwise, compute thickness and draw lines
            thickness = int(
                np.sqrt(len(path) / float((len(path)-i) + 1)) * 2.5)
            cv2.line(frame, (path[i-1][0], path[i-1][1]),
                     (path[i][0], path[i][1]), (0, 0, 255), thickness)

    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imshow("Camera image", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# shut down the camera and close all open windows
vs.stop() if video_path is None else vs.release()
cv2.destroyAllWindows()
