import cv2
import numpy as np
import imutils
import time

cap = cv2.VideoCapture("auto3.3gp")
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
car_counter = 0
inicio = 0
fin = 0
total = 0

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)

    area_pts = np.array([[215, 95], [frame.shape[1]-325, 95], [frame.shape[1]-325, 156], [215, 156]])
    area_pts2 = np.array([[640, 80], [frame.shape[1]-100, 80], [frame.shape[1]-100, 140], [640, 140]])

    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(frame, frame, mask=imAux)

    imAux2 = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux2 = cv2.drawContours(imAux2, [area_pts2], -1, (255), -1)
    image_area2 = cv2.bitwise_and(frame, frame, mask=imAux2)

    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=5)

    fgmask2 = fgbg.apply(image_area2)
    fgmask2 = cv2.morphologyEx(fgmask2, cv2.MORPH_OPEN, kernel)
    fgmask2 = cv2.morphologyEx(fgmask2, cv2.MORPH_CLOSE, kernel)
    fgmask2 = cv2.dilate(fgmask2, None, iterations=5)

    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts2 = cv2.findContours(fgmask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in cnts:
        if cv2.contourArea(cnt) > 700:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255,255), 1)

            if 260 < (x + w) < 280:
                inicio = time.time()

    for cnt in cnts2:
        if cv2.contourArea(cnt) > 700:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255,255), 1)

            if 325 < (x + w) < 635:
                fin = time.time()
                total = fin - inicio

                # Assuming a fixed distance between area_pts and area_pts2
                # You need to measure this distance in your real scenario
                distance_between_points = 100  # Example distance in pixels
                # Assuming total time is in seconds
                # You may need to convert it to appropriate time unit based on your requirement
                speed = distance_between_points / total  
                print("Velocidad: {:.2f} pixels/segundo".format(speed))
                car_counter += 1

    cv2.drawContours(frame, [area_pts], -1, (0, 255, 0), 2)
    cv2.line(frame, (267, 95), (267, 155), (255,0,0), 1)

    cv2.drawContours(frame, [area_pts2], -1, (0, 255, 0), 2)
    cv2.line(frame, (603, 80), (603, 140), (255,0,0), 1)

    cv2.imshow('Camara Web', frame)
    cv2.imshow('Automovil', fgmask)
    cv2.imshow('Automovil', fgmask2)

    k = cv2.waitKey(70) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
