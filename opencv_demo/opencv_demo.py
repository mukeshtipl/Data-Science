# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:41:35 2019

@author: MUKESHGUPTA
"""

from __future__ import print_function
import argparse
import cv2
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
print("image shape:" , image.shape)
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {}".format(image.shape[2]))

cv2.imshow("Image", image)
#cv2.waitKey(0)
cv2.imwrite("newimage.jpg", image)
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r,g, b))
corner=image[0:50,0:200]
cv2.imshow("Corner",corner)
#cv2.waitKey(0)
image[0:50,0:200]=(0,0,255)
cv2.imshow("Corner",corner)
#cv2.waitKey(0)

#create images

canvas = np.zeros((300, 300, 3), dtype = "uint8")
green = (0, 255, 0)

cv2.imshow("Canvas", canvas)


for i in range(0,300):
    for j in range(0,300):
        for k in range(0,3):
            pixel_color=np.random.randint(0,255)
            canvas[i,j,k]=pixel_color

cv2.line(canvas, (0, 100), (300, 150), green)




for i in range(0, 25):
    radius = np.random.randint(5, high = 200)
    color = np.random.randint(0, high = 256, size = (3,)).tolist()
    pt = np.random.randint(0, high = 300, size = (2,))
    cv2.circle(canvas, tuple(pt), radius, color, -1)

print(type(pt),type(color))

cv2.imshow("Canvas", canvas)


cv2.waitKey(0)
