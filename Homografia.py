#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 21:20:04 2022

@author: everto
"""

#La homagrafia nos permite mover la imagen en pocas palabras

import cv2
import numpy as np

#img = cv2.imread('cuadro.jpg')
img = cv2.imread('placa.jpg')
imG=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Obtener las dimensiones de esta imagen
#m => cantidad de filas
#n=> cantidad de columnas
#c,canales
m,n,_=img.shape

#puntos iniciales: Debo ir a la imagen y con el zoom verificar las coornadas x,y de cada esquina

#PARA EL CUADRO
#pts_src = np.array([[41,12],[193,50],[40,304],[198,271]])


#PARA LA PLACA
pts_src=np.array([[483,143],[1342,368],[537,727],[1397,980]])



# hacia donde queremos que vayan los puntos que seleccionamos
# 0,0 para x,y en la parte superior izquierda
# n, 0 para x,y en la parte superior derecha
#m, 0 en la parte inferior izquierda
# n,m en la parte inferior derecha

pts_dst=np.array([[0,0], [n,0], [0,m], [n,m]])

#Metodo para corrección de prespectivas

h,_=cv2.findHomography(pts_src, pts_dst)

im2=cv2.warpPerspective(img, h, (n,m))

cv2.imshow('original', img)
cv2.imshow('correccion', im2)

#PARA LA PLACA

