#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Based on Example Code "CameraCalibration_Tutorial.py" by @author: paavanasrinivas
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""

import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

# DEFINE CHECKERBOARD CHARACTERISTICS
chessboardSize = (7,5) # Use (Rows-1) & (Columns-1)
frameSize = (1280,720) # Dimensions of your image set
size_of_chessboard_squares_mm = 30 # Length of one square in mm

# FIND CHESSBOARD CORNERS - OBJECT AND IMAGE POINTS
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Define termination criteria
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32) # Preparing object points, like (0,0,0), (1,0,0)....,(6,5,0)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
objp = objp * size_of_chessboard_squares_mm
objpoints = [] # Creating a vector to store vectors of 3D points for each checkerboard image in real-world space
imgpoints = [] # Creating a vector to store vectors of 2D points in image place for each checkerboard image

images = glob.glob(r"C:/Users/jessi/Desktop/MAE263C/Project/arsonbot/src/camera_calibration/images/calibration_images/*.jpg") # Path to image directory

k = 1 # Initialize image naming variable
for image in images: # Extracting the path of individual images stored in a given directory
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None) # Find the chess board corners
    if ret == True: # If desired number corners detected, refine pixel coordinates and display on images of the checkerboard

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) # Refining pixel coordinates for given 2d points.
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret) # Draw and display the corners
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.show() # Using MatPlotLib to show images instead since OpenCV Gui was not compatible!

        output_filename = f"detected_corner_{k}.png"  # Image filename
        cv.imwrite(output_filename, img)
        k += 1

# CAMERA CALIBRATION
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
# Calibrate camera by passing value of known 3D points (objpoints) and corresponding pixel coordinates of detected corners (imgpoints)

print("############### Intrinsic Parameters ############### \n")
print("Camera matrix : \n") # Intrinsic Parameters
print(cameraMatrix)
print("Distance : \n")
print(dist)

print("############### Extrinsinc Parameters ############### \n")
print("Rotation Vectors (rvecs): \n") # Extrinsic Parameters
print(rvecs)
print("Translation Vectors (vecs): \n")
print(tvecs)

# UNDISTORTION
i = 1 # Initialize image name variable

for image in images:
    img = cv.imread(image) # Read image from directory
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, frameSize, 1, frameSize) # Use cv.getOptimalNewCameraMatrix
    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix) # Undistort the image
    x, y, w, h = roi # Crop image based on the region of interest (roi)
    dst = dst[y:y+h, x:x+w]
    plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB)) # Display undistorted image
    plt.show()

    output_filename = f"undistorted_{i}.png"  # Image filename
    cv.imwrite(output_filename, dst) # Save image
    i += 1

# CALCULATE RE-PROJECTION ERROR
mean_error = 0

for i in range(len(objpoints)):  # Re-projection Error
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

total_error = mean_error / len(objpoints)
print(f"Total Re-projection Error: {total_error}")








