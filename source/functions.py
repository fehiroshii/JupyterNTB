import cv2 as cv
import numpy as np


# Function that apply filters on a image
def filters(img):

    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply median blur with kernel = 9
    blur = cv.medianBlur(gray, 9)

    # Apply Gaussian blur with kernel = 9
    gauss = cv.GaussianBlur(blur, (9, 9), sigmaX=0, sigmaY=0)

    return gauss


# Returns array contaning a contour scaled
def scale_contour(cnt, scale):

    # Gets contour moments
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    # cy = int(M['m01']/M['m00'])

    # Translate contour to the origin
    cnt_norm = cnt - [cx, 0]
    # cnt_scaled = cnt_norm * scale

    # Apply the scale on the horizontal axis
    cnt_scaled = np.copy(cnt_norm)
    for i in range(len(cnt_norm)):
        cnt_scaled[i][0][0] = cnt_norm[i][0][0]*scale

    # Returns contour to its original position
    cnt_scaled = cnt_scaled + [cx, 0]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


# Auxiliary funtion to calculate the distance between two points
def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)
