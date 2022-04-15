import cv2 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


image = cv2.imread('C:/Users/Lenovo/Downloads/Face detection using opencv/images/img2.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Creating an object faces
faces= face_cascade.detectMultiScale (gray, 1.3,10)
# Drawing rectangle around the face
for(x , y,  w,  h) in faces:
  cv2.rectangle(image, (x,y) ,(x+w, y+h), (0,255,0), 2)

  # Creating two objects of interest
  roi_gray = gray[y:y+h, x:x+w]
  roi_color = image[y:y+h, x:x+w]

  #eyes = eye_cascade.detectMultiScale(roi_gray)
  #for (x_eye, y_eye, w_eye, h_eye) in eyes:
    #cv2.rectangle(roi_color,(x_eye, y_eye),(x_eye+w_eye, y_eye+h_eye), (0, 0, 255), 3)

  smile = smile_cascade.detectMultiScale(roi_gray,1.3,10)
  for (x_smile, y_smile, w_smile, h_smile) in smile: 
      cv2.rectangle(roi_color,(x_smile, y_smile),(x_smile + w_smile, y_smile + h_smile), (255, 0, 0), 3)
cv2.imshow("img",image)


cv2.waitKey()