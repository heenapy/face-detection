import numpy as np
import cv2
import dlib
# _________________________________FACE DETECTION______________________________________________________________________
face_classifier = cv2.CascadeClassifier('/home/paython/Desktop/haarcasecade/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('/home/paython/Desktop/haarcasecade/haarcascade_eye.xml')

img = cv2.imread('/home/paython/Desktop/images/face.jpeg')
cv2.imshow('real image',img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces  = face_classifier.detectMultiScale(gray,1.3,2)
# scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

if faces is ():
    print('No face found')
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('img',img)
    cv2.waitKey()
    roi_gary= gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_classifier.detectMultiScale(roi_gary)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
        cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()