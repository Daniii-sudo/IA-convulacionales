import cv2
import numpy as np
import matplotlib.pyplot as plt
imagen = cv2.imread("C:/Users/Daniel/Documents/PYTHON VS/cubos.jpg")
gris = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
bordes = cv2.Canny(gris, 100,200)
cv2.imshow ('imagen original',imagen)
cv2.imshow ('imagen color gris',gris)
cv2.imshow ('imagen gris bordes',bordes)
cv2.waitKey(0)
cv2.destroyAllWindows()