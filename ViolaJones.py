# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:47:22 2020

@author: Asaad_Salem
"""
# Face detection with Viola-Jonse algorithm


# Importing libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reading Images
face_image = cv2.imread('Screenshot.png')

# convert to grey
grey_img = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(20, 10))
plt.imshow(grey_img, cmap='gray')

# Load training file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Here is where all the magic happens
#face_detect = face_cascade.detectMultiScale(grey_img, scaleFactor = 1.1, minNeighbors=5)

# Print the number of faces found
#print ("Faces found: " , len(face_detect))
#face_detect

# Drawing rectangles around the faces
#for (x, y, w, h) in face_detect:
#    cv2.rectangle(face_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Converting back the color and plot it again
#plt.figure(figsize=(20, 10))
#plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))


faces = face_cascade.detectMultiScale(grey_img, 1.1, 6)
for (x,y,w,h) in faces:
    face_image = cv2.rectangle(face_image,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = grey_img[y:y+h, x:x+w]
    roi_color = face_image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',face_image)
cv2.waitKey(0)
cv2.destroyAllWindows()