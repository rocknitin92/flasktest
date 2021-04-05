import cv2
import numpy as np
import pandas as pd


recog = cv2.face.LBPHFaceRecognizer_create()
recog.read('recognizer/trainningData.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceDetect = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
id=1

####font = cv2.InitFont(cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img =cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray, 1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id, conf = recog.predict(gray[y:y+h,x:x+w])
        print(id,conf)
        if id==1 and conf>30 and conf<37:
            id=" Kush "
            
        elif id==2:
            id=" Pratima "
        elif id==3:
            id="nitesh"
        else:
            id="Unknown"
##        cv2.putText(img, str(id),(x,y+h),font,255,1)
        cv2.putText(img,str(id),(x+10,y),font,0.55,(255,0,0),3)
    cv2.imshow('Face',img) 
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
