import cv2
import numpy as np

def ordenar_puntos(puntos):
    n_puntos = np.concatenate([puntos[0],puntos[1],puntos[2],puntos[3]]).tolist()
    
    y_order = sorted(n_puntos,key=lambda n_puntos: n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])

    return [x1_order[0],x1_order[1],x2_order[1]]

def roi (image,ancho,alto):
    imagen_alineada = None

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray,159,255,cv2.THRESH_BINARY)
    cnts = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:1]
    for c in cnts:
        epsilon = 0.01*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)

        if len(approx) == 4:
            puntos = ordenar_puntos(approx)
            pts1 = np.float32(puntos)
            pts2 = np.float32([[0,0],[ancho,0],[0,alto],[ancho,alto]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            imagen_alineada = cv2.warpPerspective(image,M,(ancho,alto))

    return imagen_alineada

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
    ret,frame = cap.read()
    if ret == False: break
    imagen_A4 = roi(frame, ancho=720, alto=509)
    
    if imagen_A4 is not None:
      puntos = []
      imagen_HSV = cv2.cvtColor(imagen_A4,cv2.COLOR_BGR2HSV)
      verdebajo = np.array([36,14,0],np.uint8)
      verdealto = np.array([56,120,255],np.uint8)
      masVerde = cv2.inRange(imagen_HSV,verdebajo,verdealto)
      cv2.imshow('masVerde', masVerde)
      cv2.imshow('Imagen A4',imagen_A4)

    cv2.imshow('Frame',frame)
    
    k= cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()