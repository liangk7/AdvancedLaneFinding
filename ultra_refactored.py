##############################################
#   Import Statments
##############################################

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML

np.set_printoptions(threshold=np.nan)

##############################################
#   Defining Frame Class
##############################################
class Frame():
    #This class stores all of the different images in this program. Can be used later
    def __init__(self):
        self.original = None
        self.undst = None
        self.gray = None
        self.warp = None
        self.hls = None
        self.h_channel = None
        self.l_channel = None
        self.s_channel = None
        self.s_binary = None
        self.sxbinary = None
        self.color_binary = None
        self.combined_binary = None
        self.both_lines = None
        self.finalImage = None
        self.mesh = None
        self.scaled_sobel = None
        self.border = None

        self.M = None
        self.Minv = None
        self.objpoints = None
        self.imgpoints = None

        self.leftLine = Line()
        self.rightLine = Line()

        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        
##############################################
#   Defining Line Class
##############################################

class Line():
    def __init__(self):
        #Was the line detected in the last iteration?
        ##done
        self.detected = False
        
        #x values of the last n fits of the line
        self.recent_xfitted = []
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        #polynomial coefficents with world scale averaged over the last n interations
        self.best_world_fit = None

        #polynomial coefficients for the most recent fit
        self.current_fit = []

        #polynomial coefficients for most recent fits in world view
        self.current_world_fit = []

        #radius of curvature of the line in some units
        self.radius_of_curvature = None


        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype = 'float')
        
        #x values for detected line pixels
        self.allx = None

        #y values for detected line pixels
        self.ally = None

    def isLineDetected(self):
        return self.detected
    
    def updateLine(self, current_fit, current_fitx, world_fit, allx, ally):
        #updates the lines and sets the detected flag to true if we've had success in the past 5 runs, otherwise it sets it to false

        #Check if we're detected first, if not, just add the line
        if self.detected:
            #update diffs
            self.diffs = self.current_fit[-1] - current_fit

        #if the differences are too high, abort and set detected to false
        if current_fit is not None:
            if ((self.diffs[0] > 0.001) or (self.diffs[1] > 1) or (self.diffs[2] > 100)):
                #If difference is too crazy, abort and call remove line function
                self.removeLine()
            else:
                self.detected = True
            
                #update all x and all y
                self.allx = allx
                self.ally = ally

                #update polynomial coefficients
                self.current_fit.append(current_fit)
                self.current_world_fit.append(world_fit)
                self.recent_xfitted.append(current_fitx)
                
                if len(self.recent_xfitted) > 5:
                    #pop the first one if we have over 5 recent fits. Over flow protection
                    self.recent_xfitted.pop(0)
                if len(self.current_world_fit) > 5:
                    self.current_world_fit.pop(0)
                if len(self.current_fit) > 5:
                    self.current_fit.pop(0)
                  
                #average lines here and set bestx and best_fit values. Use these values for drawings
                if len(self.recent_xfitted) > 0:
                    self.bestx = np.average(self.recent_xfitted, axis = 0)
                if len(self.current_fit) > 0 :
                    self.best_fit = np.average(self.current_fit, axis = 0)
                if len(self.current_world_fit) > 0:
                    self.best_world_fit = np.average(self.current_world_fit, axis = 0)


                #Find curvature
                self.findRadius()
        #Else, no line available, set flag to not detected
        else:
            self.removeLine()



    def removeLine(self):
        #remove oldest instance of recent_xfitted and current_fit so that when it's 0 we can do sliding window method again
        if len(self.recent_xfitted) > 0:
            self.recent_xfitted.pop(0)
        if len(self.current_fit) > 0:
            self.current_fit.pop(0)
        if len(self.current_world_fit) > 0:
            self.current_world_fit.pop(0)
        if len(self.recent_xfitted) == 0 or len(self.current_fit) ==0:
            #When we have nothing left, reset and go back to default values
            self.detected = False
            self.diffs = np.array([0,0,0], dtype = 'float')
     
    def findRadius(self, height = 720):
        #Second argument is either 720 or image size
        
        #Conversions from pixel to real world
        ploty = np.linspace(0, height -1, height)
        y_eval = np.max(ploty)
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        #fit_worldSpace = np.polyfit(ym_per_pix * self.ally, xm_per_pix * self.allx, 2)
        fit_worldSpace = self.best_world_fit
        curverad = ((1 + (2*fit_worldSpace[0] * y_eval * ym_per_pix+ fit_worldSpace[1])**2)**1.5)/ np.absolute(2*fit_worldSpace[0])
        self.radius_of_curvature = curverad
        print('curverad', curverad)


##############################################
#   Defining Helper Functions
##############################################
def undistort(Frame):
    '''
    undistorts and image
    input: image, list of objpoints, and list of imgpoints
    output: undistorted image
    '''
    img = Frame.original
    objpoints = Frame.objpoints
    imgpoints = Frame.imgpoints
    img_size = (img.shape[1], img.shape[0])

    #Do camera calibration give obj points and img points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undst = cv2.undistort(img, mtx, dist, None, mtx)
    Frame.undst = undst

def visualizeImage(name, img):
    '''
    Takes in a name and image, displays it on computer
    '''
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(1)

#Transform to birds eye view
def perspectiveTransform(Frame):
    '''
    updates a photo to make it bird's eye view
    input: Original image
    output: Transformed image
    '''
    offset = 50
    undst = np.copy(Frame.undst)
    img_size = (undst.shape[1], undst.shape[0])

    #Choose source and destination points
    src = np.float32([[535, 465], [745,465], [95, 670], [1185, 670]])
    dst = np.float32([[offset, offset], [img_size[0] - offset, offset], [offset, img_size[1] - offset], [img_size[0] - offset, img_size[1] - offset]])
    M = cv2.getPerspectiveTransform(src, dst)
    Frame.M = M
    Minv = cv2.getPerspectiveTransform(dst, src)
    Frame.Minv = Minv
    warp = cv2.warpPerspective(undst, M, img_size, flags=cv2.INTER_LINEAR)
    Frame.warp = warp

    #show src area on image
    pts = np.array([[src[0][0],src[0][1]],[src[1][0],src[1][1]], [src[3][0],src[3][1]],[src[2][0],src[2][1]]], np.int32)
    pts = pts.reshape((-1,1,2))
    Frame.border = cv2.polylines(undst,[pts], True, (0,255,255))

#Sobel x plus S_gradient, put on mask
def thresholding(Frame, sobel_t_min = 50, sobel_t_max = 255, s_t_min = 200, s_t_max = 255):
    warp = Frame.warp
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    Frame.gray = gray
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) 
    abs_sobelx = np.absolute(sobelx)   
    scaled_sobel = 255*abs_sobelx/np.max(abs_sobelx)
    Frame.scaled_sobel = scaled_sobel
    
    #Sobel binary
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobel_t_min) & (scaled_sobel <= sobel_t_max)] = 1
    
    Frame.sxbinary = sxbinary
    hls = cv2.cvtColor(warp, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    Frame.hls = hls
    Frame.h_channel = h_channel
    Frame.l_channel = l_channel
    Frame.s_channel = s_channel

    #S binary
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_t_min) & (s_channel <= s_t_max)] = 1
    Frame.s_binary = s_binary

    #stack both to see the individual contributions. Green for Sobel, Blue for Saturation (HLS)
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    Frame.color_binary = color_binary

    #combine the two thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary ==1) | (sxbinary ==1)] =1
    Frame.combined_binary = combined_binary

#Training for lines
def slidingWindow(Frame):
    '''
    Takes in a thresholded image and creates sliding windows
    These midpoints will be used to determine the polynomial curve of the line
    '''
    leftLine = Frame.leftLine
    rightLine = Frame.rightLine
    
    #Take histogram of bottom part of the image
    binary_warp = Frame.combined_binary
    histogram = np.sum(binary_warp[binary_warp.shape[0]//2:,:], axis = 0)

    #Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warp, binary_warp, binary_warp))* 255

    #Find peak of left and right halves of the  histogram
    #These will be the starting points for the left and right lanes
    midpoint = np.int(histogram.shape[0] /2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #Choose number of sliding windows
    nwindows = 9
    #set height of windows
    window_height = np.int(binary_warp.shape[0]/nwindows)
    #Identify x and y positions of all nonzero pixels in the image
    nonzero = binary_warp.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    #set width of the windows margin
    margin = 100
    #set minimum number of pixels found to recenter window
    minpix = 50
    #create empty list to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    #step through the windows one by one
    for window in range(nwindows):
        #Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warp.shape[0] - (window+1)*window_height
        win_y_high = binary_warp.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        #Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low, win_y_low),(win_xleft_high, win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)
        
        #Identify nonzero pizels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        #Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        #If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    #Concatinate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    #Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #Conversions from pixel to real world
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    #Fit a second order polynomial in pixel space
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    #Fit a second order polynomial to each in real world space
    left_fit_worldSpace = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_worldSpace = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    #Visualize it
    #Generate x and y values for plotting
    ploty = np.linspace(0, binary_warp.shape[0] -1, binary_warp.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_picture = drawLines(out_img, left_fit, (100,100,0))
    both_line_picture = drawLines(left_line_picture, right_fit, (100,0,100))
    Frame.both_lines = both_line_picture


    #######################################
    #   Update Left and Right lanes
    ########################################
    leftLine.updateLine(left_fit, left_fitx, left_fit_worldSpace, leftx, lefty)
    rightLine.updateLine(right_fit, right_fitx, right_fit_worldSpace, rightx, righty)
    findDistFromCenter(Frame)


#If you already found it, much easier to find line pixels
def ezFind(Frame):
    '''
    Solves for left and right fits given that you already have left and right fits
    Skips the sliding window portion
    '''
    leftLine = Frame.leftLine
    rightLine = Frame.rightLine
    img = Frame.combined_binary
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    #uses left line and right line current_fit
    left_fit = leftLine.current_fit[-1]
    right_fit = rightLine.current_fit[-1]
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    #Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #Fit a second order polynomial in pixel space
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    #Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #Visualize result
    out_img = np.dstack((img, img, img))* 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255,0,0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0,0,255]

    #Generate a polygon to illustrate the search window area
    #Recast the x and y points into usable format for cv2.fillpoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    #Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255,0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    left_line_picture = drawLines(out_img, left_fit, (100,100,0))
    both_line_picture = drawLines(left_line_picture, right_fit, (100,0,100))
    Frame.both_lines = both_line_picture


    #######################################
    #   Update Left and Right lanes
    ########################################
    leftLine.updateLine(left_fit, left_fitx, left_fit_worldSpace, leftx, lefty)
    rightLine.updateLine(right_fit, right_fitx, right_fit_worldSpace, rightx, righty)
    findDistFromCenter(Frame)

    
def findDistFromCenter(Frame):
    leftLine = Frame.leftLine
    rightLine = Frame.rightLine

    img_height, img_width = Frame.original.shape[:2]
    
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    carPosition = img_width //2
    l_fit = leftLine.current_fit[-1]
    r_fit = rightLine.current_fit[-1]
    leftLineXint = l_fit[0]*img_height**2 + l_fit[1]*img_height + l_fit[2]
    rightLineXint = r_fit[0]*img_height**2 + r_fit[1]*img_height + r_fit[2]
    laneCenter = (leftLineXint + rightLineXint)//2
    center_dist = (carPosition - laneCenter) * xm_per_pix

    Frame.line_base_pos = center_dist


#Filling in area on image
def draw(Frame):
    '''
    draws the lines and area onto the original image
    takes in the original image and the binary warped image
    Also takes in the left and right polyfit array
    '''
    leftLine = Frame.leftLine
    rightLine = Frame.rightLine
    img = Frame.undst
    warped_img = Frame.combined_binary
    Minv = Frame.Minv

    #Fix the average!!
    left_fit = leftLine.best_fit
    right_fit = rightLine.best_fit
    
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero,  warp_zero))

    #Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #Recast the x and y ponts into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    #Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    #Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    #Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    #draw text on the image
    result = drawText(result, Frame)
    #visualizeImage('Final image', result)
    Frame.finalImage = result


def drawLines(img, fit, color):
    if fit is None:
        return img
    editImg = np.copy(img)
    ploty = np.linspace(0, img.shape[0]-1,  img.shape[0])
    plotx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
    cv2.polylines(editImg, np.int32([pts]), isClosed = False, color = color, thickness = 5)
    return editImg

def drawText(img, Frame):
    leftLine = Frame.leftLine
    rightLine = Frame.rightLine
    new_img = np.copy(img)
    h = new_img.shape[0]
    #writing down curve radius
    text = 'Curve radius :' + '{:04.2f}'.format((leftLine.radius_of_curvature + rightLine.radius_of_curvature)/2) + 'm'
    cv2.putText(new_img, text, (40,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

    #writing down center of lane
    center_dist = Frame.line_base_pos
    if center_dist > 0:
        direction = 'right'
    else:
        direction = 'left'

    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

    return new_img


##############################################
#   Main Code
##############################################

#####################################################
#   Initialize global objects
#####################################################

Frame = Frame()

##############################################
#   Calibrate Camera
##############################################

#prepare object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#Arrays to store object points and image points from all the images
Frame.objpoints = [] #3d points in real world space
Frame.imgpoints = [] # 2d points in image plane

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
        Frame.objpoints.append(objp)
        Frame.imgpoints.append(corners)

        #Draw and display the corners

        #cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #cv2.imshow('img', img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()


def visualizeAll(Frame):
    visualizeImage('original',Frame.original)
    visualizeImage('undst', Frame.undst)
    visualizeImage('gray', Frame.gray)
    visualizeImage('warp', Frame.warp)
    visualizeImage('color binary', Frame.color_binary)
    visualizeImage('both lines', Frame.both_lines)
    visualizeImage('final image', Frame.finalImage)
    visualizeImage('h channel', Frame.h_channel)
    visualizeImage('l channel', Frame.l_channel)
    visualizeImage('s channel', Frame.s_channel)
    visualizeImage('sobel', Frame.sxbinary)
    visualizeImage('s channel binary', Frame.s_binary)
    visualizeImage('mesh', Frame.mesh)

def createMesh(Frame):
    names = ['border', 'warp', 'gray', 'sobelx', 'l_channel', 's_channel', 'color_binary', 'both_lines', 'Final Image']
    display = {'undst':Frame.undst, 'gray':Frame.gray, 'warp':Frame.warp,
               'h_channel':Frame.h_channel, 'l_channel':Frame.l_channel, 's_channel':Frame.s_channel,
               'color_binary':Frame.color_binary, 'both_lines':Frame.both_lines, 'Final Image':Frame.finalImage,
               'sobelx':Frame.sxbinary, 'border':Frame.border}
    height, width = Frame.original.shape[:2]
    Frame.mesh = np.copy(Frame.original)
    offset_x = 0
    offset_y = 0
    i = 0
    for name in names:
        text = name
        img = np.copy(display[name])

        #Check and convert from grayscale to BGR
        if len(img.shape) < 3:
            try:
                #try to convert to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            except:
                img = np.dstack(( img, img, img)) * 255
        
        res = cv2.resize(img, (width//3, height //3), interpolation = cv2.INTER_LINEAR)
        #put name in the bottom middle of the image, resize it and put it on the mesh image
        cv2.putText(res,text, (res.shape[1] //2 - 50, res.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 1, cv2.LINE_AA)

        Frame.mesh[offset_y:offset_y + res.shape[0], offset_x:offset_x + res.shape[1]] = res
        
        #update i and offset values
        i += 1
        if i % 3 == 0:
            #next row
            offset_x = 0
            offset_y += height // 3
        else:
            offset_x += width // 3 

    
########################################################
#   Import Test Image
########################################################
#Import test images or video file, make into for loop later

def processImage(img):

    #img = cv2.imread('test_images/straight_lines1.jpg')

    ########################################################
    #   Process Image
    ########################################################
    #   Use functions to process image to find curves

    #Undistort original image
    #name = filename.split('/')[-1]
    #img = cv2.imread(filename)
    Frame.original = img
    undistort(Frame)

    #Change to bird's eye view
    perspectiveTransform(Frame)
    #visualizeImage('warp',warp)
    #Sobel and Sat thresh
    thresholding(Frame, sobel_t_min = 20, sobel_t_max = 255, s_t_min = 200, s_t_max = 255)
    #visualizeImage('binary_warp',binary_warp)

    #If we already have left and right lines, we can do ezFind. Otherwise, use sliding window method
    if (Frame.leftLine.isLineDetected() and Frame.rightLine.isLineDetected()):
        ezFind(Frame)
    else:
        #Use sliding window method if we haven't found anything in the last 5 checks
        slidingWindow(Frame)
        
    draw(Frame)
    #Needs to return an image for it to work
    createMesh(Frame)
    #visualizeAll(Frame)
   
    visualizeImage('mesh',Frame.mesh)

    #Change retrun statement to Frame.mesh if you want to create a mesh video
    return Frame.finalImage

def testImage(filename):
    #Run code for just one image
    name = filename.split('/')[-1]
    img = cv2.imread(filename)
    processImage(img)
    cv2.imwrite('output_images/'+name, Frame.finalImage)
    



#testImage('test_images/straight_lines1.jpg')
testImage('test_images/test4.jpg')


'''
#Video
output_file = 'video_output.mp4'
input_file = VideoFileClip('project_video.mp4')
processedClip = input_file.fl_image(processImage)
#processedClip.write_videofile(processedClip, audio = False)
%time processedClip.write_videofile(output_file, audio=False)

more
lala
'''


















        
