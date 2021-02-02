import pyautogui
import numpy as np
import cv2 as cv
import os

cars = cv.CascadeClassifier("cars.xml")
#(439, 454, 824, 356)

franklin = cv.imread("image/franklin.png")

r = cv.selectROI("Selecione o ROI", franklin, fromCenter=False)
print(r)

tracker = cv.TrackerCSRT_create()

tracker.init(franklin, r)
while True:
    os.system("rm a.png")
    #esquerda a direita, de cima para baixo, tamanho quadrado
    pyautogui.screenshot("a.png", (0,0, 1820, 920))
    img = cv.imread("a.png")
    frame_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    cv.imshow("Carros em cinza", frame_gray)
    
    carros = cars.detectMultiScale(frame_gray, 1.1, 1)

    #_, bin = cv.threshold(frame_gray, 90, 255, cv.THRESH_BINARY)
    
    #for (x,y, w, h) in carros:
        #cv.rectangle(bin, (x,y), (x+w, y+h), (0,0,255), 2)
        
    #cv.imshow("Carros em preto/branco", bin)
    
    desfoque = cv.GaussianBlur(frame_gray, (5, 5), 0)
    #cv.imshow("Desfocada", desfoque)

    #_, contours, hierarquia = cv.findContours(desfoque, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    #print(contours)
    ok, box = tracker.update(desfoque)

    if ok:
        cv.rectangle(desfoque, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)

    cv.imshow("Tracking",desfoque)
    if cv.waitKey(1) == 27:
        break
cv.destroyAllWindows()