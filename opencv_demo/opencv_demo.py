# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:41:35 2019

@author: MUKESHGUPTA
"""
# Usage : $python opencv_demo.py --image data/rooster.jpg
from __future__ import print_function
import argparse
import cv2
import numpy as np
import imutils

from matplotlib import pyplot as plt



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
print("image shape:" , image.shape)
print("image shape[:2]:" , image.shape[:2])
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {}".format(image.shape[2]))

cv2.imshow("Image", image)




cv2.waitKey(0)
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
    pt = np.random.randint(0, high = 300, size = (2,)) # size(2,) means 2 random numbers as members of array
    cv2.circle(canvas, tuple(pt), radius, color, 10) # Thickeness of line is 10, -1 is for solid fill

print(type(pt),type(color))

cv2.imshow("Canvas", canvas)



M = np.float32([[1, 0, 25], [0, 1, 50]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("Shifted Down and Right", shifted)


(h, w) = image.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by 45 Degrees", rotated)

M = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by -90 Degrees", rotated)

rotated = imutils.rotate(image, 180)
cv2.imshow("Rotated by 180 Degrees", rotated)





r = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * r))

resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resized (Width)", resized)

flipped = cv2.flip(image, 1)
cv2.imshow("Flipped Horizontally", flipped)


flipped = cv2.flip(image, 0)
cv2.imshow("Flipped Vertically", flipped)

flipped = cv2.flip(image, -1)
cv2.imshow("Flipped Horizontally & Vertically", flipped)
cropped = image[50:550 , 200:580]
cv2.imshow("Cropped", cropped)

M = np.ones(image.shape, dtype = "uint8") * 50
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)

M = np.ones(image.shape, dtype = "uint8") * 100
added = cv2.add(image, M)
cv2.imshow("Added", added)




rectangle = np.zeros((300, 300), dtype = "uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow("Rectangle", rectangle)
#cv2.waitKey(0)
circle = np.zeros((300, 300), dtype = "uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
#cv2.circle(matrix, (centrex, centrey), radius, color, solidfill)
cv2.imshow("Circle", circle)

bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)


bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)


bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)

bitwiseNot = cv2.bitwise_not(circle)
cv2.imshow("NOT", bitwiseNot)

#Masking

mask = np.zeros(image.shape[:2], dtype = "uint8")
(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75 , cY + 75), 255,-1)
cv2.imshow("Mask", mask)

masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Rectangular Mask Applied to Image", masked)



mask = np.zeros(image.shape[:2], dtype = "uint8")
cv2.circle(mask, (cX, cY), 100, 255, -1)
masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Mask", mask)
cv2.imshow("Circular Mask Applied to Image", masked)

(B, G, R) = cv2.split(image)

cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)

merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)


zeros = np.zeros(image.shape[:2], dtype = "uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))

image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image1)

hist = cv2.calcHist([image1], [0], None, [256], [0, 256])


plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()


eq = cv2.equalizeHist(image1)

cv2.imshow("Histogram Equalization", np.hstack([image1, eq])) #show both images side wise


blurred = np.hstack([
cv2.blur(image, (3, 3)),
cv2.blur(image, (5, 5)),
cv2.blur(image, (7, 7))])
cv2.imshow("Averaged", blurred)

blurred = np.hstack([
cv2.GaussianBlur(image, (3, 3), 0),
cv2.GaussianBlur(image, (5, 5), 0),
cv2.GaussianBlur(image, (7, 7), 0)])
cv2.imshow("Gaussian", blurred)


blurred = np.hstack([
cv2.medianBlur(image, 3),
cv2.medianBlur(image, 5),
cv2.medianBlur(image, 7)])
cv2.imshow("Median", blurred)

blurred = np.hstack([
cv2.bilateralFilter(image, 5, 21, 21),
cv2.bilateralFilter(image, 7, 31, 31),
cv2.bilateralFilter(image, 9, 41, 41)])
cv2.imshow("Bilateral", blurred)


cv2.waitKey(0)



