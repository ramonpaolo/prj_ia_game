import cv2 as cv

video = cv.VideoCapture("video/carros.mp4")
cars = cv.CascadeClassifier("cars.xml")

while True:
    isTrue, frame = video.read()
    cv.imshow("Carros original",frame)
    frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    carros = cars.detectMultiScale(frame_gray, 1.1, 1)

    for (x,y, a, l) in carros:
        cv.rectangle(frame_gray, (x, y), (x+a, y+l), (0,0,255), 2)
    cv.imshow("Carros ml", frame_gray)

    if cv.waitKey(27) == 27:
        break
        
cv.destroyAllWindows()