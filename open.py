import cv2 as cv

live = cv.VideoCapture(0)
live.set(3, 950)
live.set(4, 750)
HAAR_FACE = cv.CascadeClassifier("./haar_faces.xml")


while True:
    isTrue, frame = live.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frames = HAAR_FACE.detectMultiScale(gray, scaleFactor = 1.10, minNeighbors=2)
    for (x,y, w, h) in frames:
        cv.rectangle(frame, (x, y), (x+w, y+h), (29, 90, 250), thickness = 2)

    cv.imshow("live Video detected faces ", frame)
    if cv.waitKey(20) & 0xFF==ord('q'):
        break

live.release()
cv.destroyAllWindows()