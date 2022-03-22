#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:40:50 2022

@author: everto
"""

#Apartir del algoritmo de canny segmentaremos una imagen

import cv2
import numpy as np 

img=cv2.imread('iphone.jpg')

#Crear la imagen en escalas de grises, esta imagen sera usada por el algoritmo de canny

imG=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#implementamos la funcion de canny

canny=cv2.Canny(imG, 25, 125)

#Dilatación para realizar las lineas mas gruesas
kernel=np.ones((5,5), np.uint8) #Con esto si hay un pixel lo vuelvo mas grueso
bordes=cv2.dilate(canny,kernel)

#Crearemos la mascara todo los que nos interesa 255 y lo que no 0(el fondo)

_,contours,_=cv2.findContours(bordes, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#para no reescribir la variable bordes realizamos la copia

objetos=bordes.copy()

#funcion para crear nuestra mascara
 # - enviamos donde se guardara la imagen, max: arreglo con los mayores contornos, - nos permite llenar los colores de blanco y negro
cv2.drawContours(objetos,[max(contours, key=cv2.contourArea)], -1,255,thickness=-1)

#con esta funcion colocamos todo entre 0 y 1 lo que esta en cero queda en 0 y lo que esta
#en 255 queda en 1
objetos=objetos/255

#donde guardaremos la imagen segmentada
seg=np.zeros(img.shape)

#Para todas las fila, columnas y canal 0 dejando para cada canal en 0 lo que no pertenezca
#En uno lo que si pertenece

#Por ultimo: sumamos y multiplicamos porque nos permite pasar todo lo que pertenezca a la imagen
#sea de color y lo que no sea blanco
seg[:,:,0]=objetos*img[:,:,0]+255*(objetos==0)
seg[:,:,1]=objetos*img[:,:,1]+255*(objetos==0)
seg[:,:,2]=objetos*img[:,:,2]+255*(objetos==0)

#Transformamos todos los datos a enteros
seg=np.uint8(seg)

cv2.imshow('original', img)
cv2.imshow('segmentada', seg)