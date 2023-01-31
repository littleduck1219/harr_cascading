import cv2

image = cv2.imread("./people1.jpg")
print(image.shape)
image = cv2.resize(image, (800, 600))
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
face_detection = face_detector.detectMultiScale(img_gray, scaleFactor=1.09, minNeighbors=10)
for (x, y, w, h) in face_detection:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

eye_detector = cv2.CascadeClassifier("./haarcascade_eye.xml")
eye_detection = eye_detector.detectMultiScale(img_gray, scaleFactor=1.09, minNeighbors=10, maxSize=(72, 72))
for (x, y, w, h) in eye_detection:
    print(w, h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

for (x, y, w, h) in car_detection:
    print(w, h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("people", image)
cv2.waitKey(0)
