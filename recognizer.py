import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("trainingdata.yml")
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#font=cv2.InitFont(cv2.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while 1:
    id=0
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if id==1:
            id="hi revanth"
        elif id==2:
            id="hi sanju"
        else:
            id="unknown"
        cv2.putText(img,str(id),(x,y+h),font,1.5,(51,204,255),1)
    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()


