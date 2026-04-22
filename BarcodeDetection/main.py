# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "path to the image file")
args = vars(ap.parse_args())


# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4
ddepth = cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
# Barcode lines are mostly vertical
# So vertical edges (gradX) are strong
# Horizontal edges (gradY) are weaker/noise

# subtract the y-gradient from the x-gradient
# Strong vertical edge → large value in gradX
# Subtracting gradY removes unwanted edges
gradient = cv2.subtract(gradX, gradY)

# Gradient values can be negative or large
# This converts them to:
# absolute values
# range: 0–255
# Now it’s a normal grayscale image again
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
# Averages neighboring pixels
# Smooths small variations
# Barcode = many thin lines
# Blurring merges them into a solid block


(_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
# Pixels > 200 → become 255 (white)
# Pixels ≤ 200 → become 0 (black)
# Bright regions (likely barcode) → white
# Everything else → black

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
# Barcodes are wider than they are tall
# So we use a wide kernel to connect vertical lines horizontally


closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# Closing = dilation followed by erosion

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
# Removes small white noise
# Breaks weak connections
# Smooths boundaries
# Thin noise disappears
# Only strong regions remain


closed = cv2.dilate(closed, None, iterations = 4)
# Grows remaining shapes back
# Restores main object size



# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
# Finds boundaries of white regions in a binary image
# White = object
# Black = background
# So in this case, The barcode blob (white) becomes a contour


cnts = imutils.grab_contours(cnts)
# extracts the actual contour list safely


c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
# Fits the smallest rectangle (any angle) around the contour
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect) # gives 4 corners of the rectangle
# box = np.int8(box)
box = box.astype(int) # convert float coordinates to integers (required for drawing)
# draw a bounding box arounded the detected barcode and display the
# image
# cv2.drawContours(image, [box], -1, (255, 0, 0), 3)
cv2.drawContours(image, [box], -1, (255, 0, 0), 3)

# cv2.drawContours(
#     image,
#     [np.array([[100, 298],
#                [200, 296],
#                [206, 141],
#                [100, 144]])],
#     -1,
#     (255, 0, 0),
#     3
# )

cv2.imshow("Image", image)
cv2.waitKey(0)


