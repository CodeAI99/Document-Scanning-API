# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 13:53:56 2019

@author: anshu
"""

from skimage.filters import threshold_local

import argparse
import cv2
import imutils
import numpy as np

import time
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--image")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0] / 300.0
original = image.copy()
image = imutils.resize(image, height = 300)
 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edged_image = cv2.Canny(gray_image, 0, 200)
 
cv2.imshow("Image", image)
cv2.imshow("Edged Image", edged_image)
end = time.time()
print("Execution Time For Edge Detection and Cropping -- %s seconds " % (end - start))
start2 = time.time()
cont = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key = cv2.contourArea, reverse = True)
 
for x in cont:
	a = cv2.arcLength(x, True)
	approx = cv2.approxPolyDP(x, 0.02 * a, True)
 
	if len(approx) == 4:
		screencont = approx
		break
 
cv2.drawContours(image, [screencont], -1, (0, 255, 0), 2)
cv2.imshow("Outline Image", image)

cv2.imwrite("edge_detection.jpg", image)

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped


transformed = four_point_transform(original, screencont.reshape(4, 2) * ratio)
 
transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
T = threshold_local(transformed, 21, offset = 15, method = "gaussian")
transformed = (transformed > T).astype("uint8") * 255
sizetransformed = imutils.resize(transformed, height = 500)
cv2.imshow("Scanned", sizetransformed)
cv2.imwrite("perspective_transformation.jpg", sizetransformed)
end2 = time.time()
print("Execution Time For Perspective Transformation -- %s seconds " % (end2 - start2))
cv2.waitKey(0)
