##############################################
#   Advanced Lane Lines
##############################################
'''
By: Sean Pan
10/10/2017
'''


##############################################
#   Import Statments
##############################################

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
#import imageio
#imageio.plugins.ffmpeg.download()
#from moviepy.editor import VideoFileClip

#from IPython.display import HTML


np.set_printoptions(threshold=np.nan)
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

        #polynomial coefficients for the most recent fit
        self.current_fit = []

        #radius of curvature of the line in some units
        self.radius_of_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype = 'float')
        
        #x values for detected line pixels
        self.allx = None

        #y values for detected line pixels
        self.ally = None

    def isLineDetected(self):
        return self.detected
    
    def updateLine(self, current_fit, current_fitx, allx, ally):
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
                self.recent_xfitted.append(current_fitx)
                
                if len(self.recent_xfitted) > 5:
                    #pop the first one if we have over 5 recent fits. Over flow protection
                    self.recent_xfitted.pop(0)
                if len(self.current_fit) > 5:
                    self.current_fit.pop(0)
                  
                #average lines here and set bestx and best_fit values. Use these values for drawings
                if len(self.recent_xfitted) > 0:
                    self.bestx = np.average(self.recent_xfitted, axis = 0)
                if len(self.current_fit) > 0 :
                    self.best_fit = np.average(self.current_fit, axis = 0)

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
        if len(self.recent_xfitted) == 0 or len(current_fit) ==0:
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
        
        fit_worldSpace = np.polyfit(ym_per_pix * self.ally, xm_per_pix * self.allx, 2)

        curverad = ((1 + (2*fit_worldSpace[0] * y_eval * ym_per_pix+ fit_worldSpace[1])**2)**1.5)/ np.absolute(2*fit_worldSpace[0])
        self.radius_of_curvature = curverad
    

    
    
        
        
        


##############################################
#   Defining Functions
##############################################
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
    undst = cv2.undistort(img, mtx, dist, None, mtx)

    return undst

def visualizeUndistort(img, undst):
    
    cv2.imshow('img', img)
    cv2.imshow('undst', undst)
   
def visualizeImage(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
        
#Transform to birds eye view
def perspectiveTransform(undst):
    '''
    updates a photo to make it bird's eye view
    input: Original image
    output: Transformed image
    '''
    offset = 50
    img_size = (undst.shape[1], undst.shape[0])

    #find source and destination points
    #####################################################
    #src = np.float32([[630, 425], [650,425], [1060, 670], [250, 670]]) # find points from my mask?
    #630, 425 650, 425 250, 670, 1060, 670
    #src = np.float32([[575, 465], [710,465], [250, 670], [1050, 670]])
    src = np.float32([[540, 465], [740,465], [95, 670], [1185, 670]])
    #####################################################
    dst = np.float32([[offset, offset], [img_size[0] - offset, offset], [offset, img_size[1] - offset], [img_size[0] - offset, img_size[1] - offset]])
    #dst = np.float32([[offset, 0], [img_size[0] - offset , 0], [offset, img_size[1]], [img_size[0] - offset, img_size[1]]]) 
    #compute perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #warp image
    warp = cv2.warpPerspective(undst, M, img_size, flags=cv2.INTER_LINEAR)

    return (warp, Minv)    

#Sobel x plus S_gradient, put on mask
def thresholding(img, sobel_t_min = 50, sobel_t_max = 255, s_t_min = 200, s_t_max = 255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    visualizeImage('gray', gray)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) 
    abs_sobelx = np.absolute(sobelx)   
    scaled_sobel = abs_sobelx    # This doesn't work np.uint8(255*abs_sobelx/np.max(abs_sobelx))
   
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobel_t_min) & (scaled_sobel <= sobel_t_max)] = 1

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    ##Debugging
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    visualizeImage('hls', hls)
    visualizeImage('h_channel', h_channel)
    visualizeImage('l_channel', l_channel)
    visualizeImage('s_channel', s_channel)

    #S binary
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_t_min) & (s_channel <= s_t_max)] = 1
    #print('s_binary', s_binary)
    visualizeImage('s_binary', s_binary)

    #L binary
    #l_binary = np.zeros_like(l_channel)
    #l_binary[(l_channel >= s_t_min) & (l_channel <= s_t_max)] = 1
    #visualizeImage('l_channel', l_binary)

    #stack both to see the individual contributions. Green for Sobel, Blue for Saturation (HLS)
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255


    #color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, l_binary))*255
    #combine the two thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary ==1) | (sxbinary ==1)] =1
    visualizeImage('color_binary', color_binary)

    return combined_binary

#Training for lines

def slidingWindow(binary_warp, leftLine, rightLine):
    '''
    Takes in a thresholded image and creates sliding windows
    These midpoints will be used to determine the polynomial curve of the line
    '''
    #Take histogram of bottom part of the image
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
    plt.imshow(out_img)

    #visualizeImage('out_img', out_img)
    plt.plot(left_fitx, ploty, color = 'yellow')
    plt.plot(right_fitx, ploty, color = 'yellow')
    plt.xlim(0, 1289)
    plt.ylim(720, 0)

    left_line_picture = drawLines(out_img, left_fit, (100,100,0))
    both_line_picture = drawLines(left_line_picture, right_fit, (100,0,100))
    visualizeImage('both lines', both_line_picture)
    
    

    #######################################
    #   Update Left and Right lanes
    ########################################
    leftLine.updateLine(left_fit, left_fitx, leftx, lefty)
    rightLine.updateLine(right_fit, right_fitx, rightx, righty)
    findDistFromCenter(leftLine, rightLine, img_width = 1280, img_height = 720)
    
    return (left_fit, right_fit, left_fit_worldSpace, right_fit_worldSpace) #Return numpy array

#If you already found it, much easier to find line pixels
def ezFind(img, leftLine, rightLine):
    '''
    Solves for left and right fits given that you already have left and right fits
    Skips the sliding window portion
    '''
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
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color = 'yellow')
    plt.plot(right_fitx, ploty, color = 'yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    
    #######################################
    #   Update Left and Right lanes
    ########################################
    leftLine.updateLine(left_fit, left_fitx, leftx, lefty)
    rightLine.updateLine(right_fit, right_fitx, rightx, righty)
    findDistFromCenter(leftLine, rightLine, img_width = 1280, img_height = 720)
   
def findDistFromCenter(leftLine, rightLine, img_width = 1280, img_height = 720):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    carPosition = img_width //2
    l_fit = leftLine.current_fit[-1]
    r_fit = rightLine.current_fit[-1]
    leftLineXint = l_fit[0]*img_height**2 + l_fit[1]*img_height + l_fit[2]
    rightLineXint = r_fit[0]*img_height**2 + r_fit[1]*img_height + r_fit[2]
    laneCenter = (leftLineXint + rightLineXint)//2
    center_dist = (carPosition - laneCenter) * xm_per_pix
    leftLine.line_base_pos = center_dist #repeat for redundancy
    rightLine.line_base_pos = center_dist

    
#Filling in area on image
def draw(img, warped_img, Minv, leftLine, rightLine):
    '''
    draws the lines and area onto the original image
    takes in the original image and the binary warped image
    Also takes in the left and right polyfit array
    '''

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

    #visualizeImage('before warp', color_warp)
    #Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    #Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    #plt.imshow(result)

    #draw text on the image
    result = drawText(result, leftLine, rightLine)
    visualizeImage('Final image', result)
    return result

def drawLines(img, fit, color):
    if fit is None:
        return img
    editImg = np.copy(img)
    ploty = np.linspace(0, img.shape[0]-1,  img.shape[0])
    plotx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
    cv2.polylines(editImg, np.int32([pts]), isClosed = False, color = color, thickness = 5)
    return editImg

def drawText(img, leftLine, rightLine):
    new_img = np.copy(img)
    h = new_img.shape[0]
    #writing down curve radius
    text = 'Curve radius :' + '{:04.2f}'.format((leftLine.radius_of_curvature + rightLine.radius_of_curvature)/2) + 'm'
    cv2.putText(new_img, text, (40,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

    #writing down center of lane
    center_dist = leftLine.line_base_pos
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

##############################################
#   Calibrate Camera
##############################################

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
        #cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #cv2.imshow('img', img)
        #cv2.waitKey(500)

cv2.destroyAllWindows()

#####################################################
#   Create Lines
#####################################################
leftLine = Line()
rightLine = Line()

########################################################
#   Import Test Image
########################################################
#Import test images or video file, make into for loop later

def testImage(filename):
#Run code for just one image
    #img = cv2.imread('test_images/straight_lines1.jpg')

    ########################################################
    #   Process Image
    ########################################################
    #   Use functions to process image to find curves

    #Undistort original image
    name = filename.split('/')[-1]
    img = cv2.imread(filename)
    undst = undistort(img, objpoints, imgpoints)

    #Change to bird's eye view
    warp, Minv = perspectiveTransform(undst)
    visualizeImage('warp',warp)
    #Sobel and Sat thresh
    binary_warp = thresholding(warp, sobel_t_min = 40, sobel_t_max = 255, s_t_min = 200, s_t_max = 255)
    visualizeImage('binary_warp',binary_warp)

    #If we already have left and right lines, we can do ezFind. Otherwise, use sliding window method
    if (leftLine.isLineDetected() and rightLine.isLineDetected()):
        ezFind(binary_warp, leftLine, rightLine)
    else:
        #Use sliding window method if we haven't found anything in the last 5 checks
        slidingWindow(binary_warp, leftLine, rightLine)
        
    finalImage = draw(undst, binary_warp, Minv, leftLine, rightLine)
    cv2.imwrite('output_images/'+name, finalImage)
    
'''
def processVideo(video):
    clip = VideoFileClip(video)
    processedClip = clip.fl_image(testImage)
    %time processedClip.write_videofile(processedClip, audio = False)
'''

    
testImage('test_images/test1.jpg') 
'''
output_file = 'video_output.mp4'
input_file = VideoFileClip('project_video.mp4')
processedClip = input_file.fl_image(testImage)
processedClip.write_videofile(processedClip, audio = False)

'''


















