import cv2

face_cascades  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces  = face_cascades.detectMultiScale(gray, 1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(255,255,0),5)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    
    
        #eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)
        #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(0,0,255),2)
            
        smile = smile_cascade.detectMultiScale(roi_gray,1.5,5)
        for (x_smile, y_smile, w_smile, h_smile) in smile: 
            cv2.rectangle(roi_color,(x_smile, y_smile),(x_smile + w_smile, y_smile + h_smile), (255, 0, 0), 3)
             
    cv2.imshow('img', frame)
    k=cv2.waitKey(30)
    if k == 27:
        break
    
camera.release()
cv2.destroyAllWindows()