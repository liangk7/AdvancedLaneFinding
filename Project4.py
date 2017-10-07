#Advanced Lane Lines
#import statements
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


#Calibrate Camera

#prepare object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#Arrays to store object points and image points from all the images
objpoints = [] #3d points in real world space
imgpoints = [] # 2d points in image plane

#Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

#Step through the lsit and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #find corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    #If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        #Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()


def undistort(img, objpoints, imgpoints):
    '''
    undistorts and image
    input: image, list of objpoints, and list of imgpoints
    output: undistorted image
    '''
    img_size = (img.shape[1], img.shape[0])
    #print(img_size)

    #Do camera calibration give obj points and img points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    #visualize undistortion
    #visualizeUndistort(img, dst)
    #return undistorted image
    return dst

def visualizeUndistort(img, dst):
    f, (ax1, ax2) = plt.subplots(1,2, figsize = (20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image')




#Transform to birds eye view
def perspectiveTransform(img):
    '''
    updates a photo to make it bird's eye view
    input: Original image
    output: Transformed image
    '''
    offset = 100
    img_size = (img.shape[1], img.shape[0])

    #find source and destination points
    #####################################################
    src = np.float32([]) # find points from my mask?
    #####################################################
    dst = np.float32([[offset, offset], [img_size[0] - offset, offset], [img_size[0] - offset, img_size[1] - offset], [offset, img_size[1] - offset]]) 
    #compute perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)
    #warp image
    warp = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warp
    

#Sobel x plus S_gradient, put on mask
def thresholding(img, sobel_t_min = 20, sobel_t_max = 100, s_t_min = 170, s_t_max = 255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobel_t_min) & (scaled_sobel <= sobel_t_max)] = 1

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.seros_like(s_channel)
    s_binary[(s_channel >= s_t_min) & (s_channel <= s_t_max)] = 1

    #stack both to see the individual contributions. Green for Sobel, Blue for Saturation (HLS)
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    #combine the two thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary ==1) | (sxbinary ==1)] =1

    return combined_binary
    
    
#Training for lines

#Creating polyfit of left and right lanes

#Calculating radius

#Drawing line on image

#Filling in area on image


