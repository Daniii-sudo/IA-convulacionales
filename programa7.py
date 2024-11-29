import cv2
import numpy as np


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()  
    if not ret:
        break

    # color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # rangos para el color rojo en HSV
    rojo_bajo1 = np.array([0, 120, 70], np.uint8)
    rojo_alto1 = np.array([10, 255, 255], np.uint8)
    rojo_bajo2 = np.array([170, 120, 70], np.uint8)
    rojo_alto2 = np.array([180, 255, 255], np.uint8)


    # se crean mascaras para rangos de rojo
    mask1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
    mask2 = cv2.inRange(hsv, rojo_bajo2, rojo_alto2)
    mask = cv2.add(mask1, mask2)

    # detectar bordes
    canny = cv2.Canny(mask, 10, 150)
    canny = cv2.dilate(canny, None, iterations=5)
    canny = cv2.erode(canny, None, iterations=5)

    # contornos en la mascara
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            cv2.putText(frame, 'Triangulo', (x, y - 5), 1, 1, (0, 255, 0), 1)

        if len(approx) == 4:
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:  # margen para errores
                cv2.putText(frame, 'Cuadrado', (x, y - 5), 1, 1, (0, 255, 0), 1)
            else:
                cv2.putText(frame, 'Rectangulo', (x, y - 5), 1, 1, (0, 255, 0), 1)

        if len(approx) == 5:
            cv2.putText(frame, 'Pentagono', (x, y - 5), 1, 1, (0, 255, 0), 1)

        if len(approx) == 6:
            cv2.putText(frame, 'Hexagono', (x, y - 5), 1, 1, (0, 255, 0), 1)

        if len(approx) > 10:
            cv2.putText(frame, 'Circulo', (x, y - 5), 1, 1, (0, 255, 0), 1)

        # contorno detectado
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

    
    cv2.imshow('Formas Detectadas', frame)

    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()