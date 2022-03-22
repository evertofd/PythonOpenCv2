#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:06:56 2022

@author: everto
"""

import cv2
import numpy as np
from scipy import ndimage

#lIBRERIA PARA CARGAR EL VIDEO
cap=cv2.VideoCapture('trafico3.mp4')
#CONTAMOS LA CANTIDAD DE FRAME QUE EXISTEN EN EL VIDEO DE MANERA AUTOMATICA
length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video=[]
for i in range (0,length):
    #ret es boleano e indica si es parte de los frame que necesitamos y es valido para leer
    ret,frame=cap.read()
    if ret:
        video.append(frame)
        
#Nuestro frame sera tomado pixel a pixel y se realziara una suma para ello
#Veo sus dimensiones son el metodo shape
suma=np.zeros(video[0].shape)

for frameI in video:
    suma=suma+frameI
    
#Para obtener el modelo de fondo dividimos entre la cantidad de imagenes

modeloFondo=np.uint8(suma/len(video))


for frameI in video:
    #SOLO PARA VISUALIZAR
    #frameI=video[50]
    #Calcular la diferencia entre frame actual y nuestro modelo de fondo
    diferencia=cv2.absdiff(frameI, modeloFondo).astype('uint8')
    
    #Para disminuir el ruido en la imagen
    diferencia=cv2.GaussianBlur(diferencia,(11,11), 0)
    #A escalas de grices para calcular con solo un color
    diferencia=cv2.cvtColor(diferencia,cv2.COLOR_BGR2GRAY)
    
    #COmparar con un valor de umbral para detectar objetos en la imagen
    umbral,_=cv2.threshold(diferencia, 0, 255, cv2.THRESH_OTSU)
    
    #Si es mayor es 1 qque significa 1 objeto si es 0 es el fondo
    bina=255*np.uint8(diferencia>umbral)
    
    
    bina=255*np.uint8(ndimage.binary_fill_holes(bina))
    
    #Para visualizar estos en la iamgen original se usara el bondiboll recto y cara punto sera rodiado 
    _,contours,_=cv2.findContours(bina, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for i in contours:
        rect=cv2.boundingRect(i)
        #x,y,ancho,altura
        x,y,w,h=rect
        
        #Para simplificar el area de captura
        if cv2.contourArea(i)>200:#Este 200 es un area determinada donde quiero detertar las imagenes
        #Los dibujamos
            cv2.rectangle(frameI, (x,y), (x+w , y+h), (0,255,0),2)


    
    cv2.imshow("Diferencia", frameI)
#cv2.imshow("Fondo", modeloFondo)

# 0 funciona para imagenes y 500 para videos
    cv2.waitKey(500)