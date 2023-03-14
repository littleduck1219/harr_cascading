import cv2

# load 05.image
image = cv2.imread("./family.jpg")
print(image.shape)

# resize
image = cv2.resize(image, (800, 600))
print(image.shape)

# convert RGB -> GRAY
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # opencv RGB = BGR

# cv2.imshow("people", img_gray)
# cv2.waitKey(0)

face_detect = cv2.CascadeClassifier("./haarcascade_frontalface_default.02.voc_to_yolo")
detections = face_detect.detectMultiScale(img_gray, scaleFactor=3, minNeighbors=1)  # 배경에서 이미지를 찾아서 크고 작게 조절할 수있습니다. 감지 물체 주위의 후보 박스들 중에서 최적의 박스를 선택
print(detections)
print(len(detections))

for (x, y, w, h) in detections:
    print(x, y, w, h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("people", image)
cv2.waitKey(0)

