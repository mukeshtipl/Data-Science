# USAGE
#python scan_to_pdf.py -i data/receipt.jpg

# import the necessary packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from fpdf import FPDF
from PIL import Image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
#gray = cv2.GaussianBlur(gray, (3, 3), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
#cv2.waitKey(0)
#cv2.imshow("Edged", gray)
#cv2.waitKey(0)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()




#  perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.dilate(edged, None, iterations=5)
edged = cv2.erode(edged, None, iterations=1)

cv2.imshow("Edged with dilation and erode", edged)
cv2.waitKey(0)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#T = threshold_local(warped, 11, offset = 10, method = "gaussian")
T = threshold_local(warped, 11, offset = 2, method = "mean")




'''
sobelX = cv2.Sobel(warped, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(warped, cv2.CV_64F, 0, 1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.waitKey(0)
'''



warped = (warped > T).astype("uint8") * 255
#warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

# show the original and scanned images
print(warped.shape)




print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))


#need to add check if rotation is required or not


print("warped.size",warped.size,warped.shape)
width, height=warped.shape
if width <height:
    nimage=imutils.rotate_bound(warped,-90)
else:
    nimage=warped



'''
#Adjust contrast and brightness
alpha=2
beta=3
for y in range(nimage.shape[0]):
    for x in range(nimage.shape[1]):
        nimage[y,x]= np.clip(alpha*nimage[y,x] + beta, 0, 255)
        
'''
cv2.imshow("Final" ,imutils.resize(nimage, height = 650))

cv2.imwrite("data/scanned_image.jpg",nimage)
scanned_image=Image.open("data/scanned_image.jpg")
width, height=scanned_image.size
pdf = FPDF(unit = "pt", format = [width, height])
pdf.add_page()
pdf.image("data/scanned_image.jpg",0,0)
pdf.output("data/scanned_image.pdf", "F")
cv2.waitKey(0)