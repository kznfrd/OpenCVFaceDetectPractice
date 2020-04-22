import cv2

cascade_path = "./haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)
color = (255, 255, 255)
cap = cv2.VideoCapture(0)

scale_factor = 1.1
min_neighbors = 2
min_size = (2, 2)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    rect = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    if len(rect) > 0:
        for x, y, w, h in rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color)
    
    cv2.imshow('detected', frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
