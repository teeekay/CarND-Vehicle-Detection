import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import scipy.signal

'''
information needed for undistort
'''
dist_pickle = pickle.load( open( "./camera_cal/cal_dist_pickle.p", "rb" ) )
undistort_mtx = dist_pickle["mtx"]
undistort_dist = dist_pickle["dist"]

'''
func: undistort
removes intrinsic camera distortion effects
leaves black in frame (does not crop) outside visible edges of corrrected image
'''
def camera_undistort(img):
    # Use the OpenCV undistort() function to remove distortion
    h,  w = img.shape[:2]
    new_mtx, roi=cv2.getOptimalNewCameraMatrix(undistort_mtx,undistort_dist,(w,h),1,(w,h))
    undist = cv2.undistort(img, undistort_mtx, undistort_dist, None, new_mtx)
    return undist

undistort_src = np.float32([[59,63],
                           [1244, 63],
                           [1244, 673],
                           [59, 673]])

undistort_dst = np.float32([[0, 0],
                           [1279, 0],
                           [1279,719],
                           [0,719]])
def undistort_crop(img):
    # Given src and dst points, calculate the perspective transform matrix
    undistort_M = cv2.getPerspectiveTransform(undistort_src, undistort_dst)
    img_size =(1280,720)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, undistort_M, img_size)
    return warped

'''
information needed for birdseye_transform
'''
birdseye_src = np.float32([[617, 450],
                           [711, 450],
                           [988, 626],
                           [359, 626]])

birdseye_dst = np.float32([[260, 640],
                           [460, 640],
                           [460,1260],
                           [260,1260]])
'''
func: birdseye_transform()
transform image from perspective view to overhead (birdseye) view
'''
def birdseye_transform(img):
    # Given src and dst points, calculate the perspective transform matrix
    birdseye_M = cv2.getPerspectiveTransform(birdseye_src, birdseye_dst)
    img_size = (img.shape[1],img.shape[0])
    img_size =(720,1280)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, birdseye_M, img_size)
    return warped
'''
func: birdseye_untransform()
transform image from overhead (birdseye) to perspective view view
'''
def birdseye_untransform(img):
    # Given src and dst points, calculate the perspective transform matrix
    birdseye_M_inv = cv2.getPerspectiveTransform(birdseye_dst, birdseye_src)
    #img_size = (img.shape[1],img.shape[0])
    img_size =(1280,720)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, birdseye_M_inv, img_size)
    return warped


'''
sobel_thresh
input: single plane image
choice to apply sobel in x, y or both and using different derivative or filter kernel size
'''
X_DIR = 1
Y_DIR = 2
MAGNITUDE = 4+2+1
ANGLE = 8+2+1

def sobel_thresh(img, orient=X_DIR, thresh_min=0, thresh_max=255, ksize=3, deriv=1):
    if img.ndim != 2:
        print("Error in sobel_thresh: img is supposed to have 2 dimensions, but has {} dimensions".format(img.ndim))
        return False, np.array([0])
    X_DIR = 1
    Y_DIR = 2
    MAGNITUDE = 4+2+1
    ANGLE = 8+2+1


    # Take the derivative in x or y
    if orient & X_DIR == X_DIR:
        sobel = cv2.Sobel(img, cv2.CV_64F, 1*deriv, 0, ksize=ksize)
        #3) Take the absolute value of the derivative or gradient
        abs_sobelx = np.absolute(sobel)
        abs_sobel = abs_sobelx
    if orient & Y_DIR == Y_DIR:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1*deriv, ksize=ksize)
        #Take the absolute value of the derivative or gradient
        abs_sobely = np.absolute(sobel)
        abs_sobel = abs_sobely

    #calculate magnitude or angle and scale output to range 0-255
    if orient == MAGNITUDE:
        magnitude = np.sqrt(abs_sobelx**2 + abs_sobely**2)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    elif orient == ANGLE:
        angle = np.arctan2(abs_sobely, abs_sobelx)
        # scale to pi/2 = 255 - the direction is important
        print("Min Angular value is {} and Max value is {}.".format(np.min(angle/np.pi),np.max(angle/np.pi)))
        scaled_sobel = np.uint8(255*angle/(np.pi/2))
    else:
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # 6) Return this mask as your binary_output image
    return True, sbinary

'''
experimental function to find lines
second derivative down lines (generally in y direction should be close to zero)

'''
def sobel_grad(img, ksize=(5,3), deriv=(1,2)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, deriv[0], 0, ksize=ksize[0])
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, deriv[1], ksize=ksize[1])
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx)
    sobel = np.dstack((np.zeros_like(sobelx),magnitude, angle))
    return sobel

'''
custom line detection kernels built to find lines that are 4 to 6 pixels wide and generally vertical
kernel1 to remove noise smaller than 3*3
kernel2 for 10 pixel wide solid line
'''
kernel1 = np.array([[1,1,1],[1,1,1],[1,1,1]])

kernel2 = np.array([[0,-1,1,0,0,0,0,0,0,0,0,1,-1,0],
                    [0,-1,1,0,0,0,0,0,0,0,0,1,-1,0],
                    [0,-1,1,0,0,0,0,0,0,0,0,1,-1,0],
                    [0,-1,1,0,0,0,0,0,0,0,0,1,-1,0],
                    [0,-1,1,0,0,0,0,0,0,0,0,1,-1,0]])

def line_pipeline(img, convolve=False):
    #remove camera distortion
    tru_img = camera_undistort(img)
    #switch to overhead view
    birdseye_img = birdseye_transform(tru_img)

    # pull out channels to be used for thresholding
    b_img = birdseye_img[:,:,0]
    g_img = birdseye_img[:,:,1]
    r_img = birdseye_img[:,:,2]
    hsv_img = cv2.cvtColor(birdseye_img, cv2.COLOR_BGR2HSV)
    h_img = hsv_img[:,:,0]
    s_img = hsv_img[:,:,1]
    v_img = hsv_img[:,:,2]
    HLS_img = cv2.cvtColor(birdseye_img, cv2.COLOR_BGR2HLS)
    H_img = HLS_img[:,:,0]
    L_img = HLS_img[:,:,1]
    S_img = HLS_img[:,:,2]
    YCrCb_img = cv2.cvtColor(birdseye_img,cv2.COLOR_BGR2YCR_CB)
    Y_img = YCrCb_img[:,:,0]
    CR_img = YCrCb_img[:,:,1]
    CB_img = YCrCb_img[:,:,2]

    sobel_inp  = np.copy(v_img)

    sgrad_img = sobel_grad(sobel_inp)
    sobel_mag = sgrad_img[:,:,1]
    sobel_thresh = np.zeros_like(sobel_inp, dtype=np.uint8)
    sobel_thresh[sobel_mag>1750] = 1
    #thresh_mag_and_h_and_L = np.zeros_like(sobel_mag)

    rc = np.zeros_like(sobel_inp, dtype=np.uint8)

    rc[ (sobel_thresh>0)|
       (v_img>220) |
       ((h_img>=19) &(h_img<=24)) |
       ((H_img>17)&(H_img<45)&(L_img>140)&(L_img<180)&(S_img>80))|
       (L_img > 220) |
       (Y_img>200)|
       ((CR_img>142)&(CR_img<170))|
       ((CB_img>30)&(CB_img<110)) |
       ((r_img>225)&(g_img>180)&(b_img<170))]=1

    # Do a little noise removal
    conv = scipy.signal.convolve2d(rc, kernel1, mode='same')
    rc = np.zeros_like(conv, dtype=np.uint8)
    rc[conv>6]=1

    # get rid of shadows and other pathches with a convolution
    if convolve == True:
        #print("using convolution", end = '')
        rc = scipy.signal.convolve2d(rc, kernel2, mode='same')
        conv = np.zeros_like(rc, dtype=np.uint8)
        conv[rc>0]=1
        rc = conv

    return rc, tru_img, birdseye_img

'''
started out as a place to stash an image, morphed into storage for
data between frames
'''
class data_storage(): #binary_image():
    def __init__(self):
        self.new_image = np.array([0])
        self.old_image = np.array([0])
        #displacement from last image
        self.displacement = []
        self.frame = 0
        self.fps = 25 #default non null value
        # was the line detected in the last iteration?
        self.linesdetected = False
        #polynomial coefficients
        self.avg_lft_fit = np.array([0.,0.,0.])
        self.avg_rgt_fit = np.array([0.,0.,0.])
        self.left_fita = []
        self.left_fitb = []
        self.left_fitc = []
        self.right_fita = []
        self.right_fitb = []
        self.right_fitc = []
        #radius of curvature of the line in some units
        self.curvatureL = []
        self.curvatureR = []
        #count of frames where generally non parallel lines collected
        self.bad_frames = 0
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
    def add_new_image(self, bin_img):
#        print("Adding new image to storage.")
        self.old_image = np.copy(self.new_image)
        self.new_image = np.copy(bin_img)
    def add_fps(self, fps):
        self.fps = fps
    def add_displacement(self, displacement):
        self.displacement = self.displacement[-3:]
        self.displacement.append(displacement)
    def add_curvature(self, curvatureL, curvatureR):
        self.curvatureL = self.curvatureL[-7:]
        self.curvatureL.append(curvatureL)
        self.curvatureR = self.curvatureR[-7:]
        self.curvatureR.append(curvatureR)
    def set_lines_detected(self, detected):
        #should be a boolean
        self.linesdetected = detected
    def save_fitx(self, left_fitx,right_fitx):
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.bad_frames = 0
    def save_fit(self, left_fit, right_fit):
        self.left_fita = self.left_fita[-3:]
        self.left_fitb = self.left_fitb[-3:]
        self.left_fitc = self.left_fitc[-3:]
        self.left_fita.append(left_fit[0])
        self.left_fitb.append(left_fit[1])
        self.left_fitc.append(left_fit[2])
        self.right_fita = self.right_fita[-3:]
        self.right_fitb = self.right_fitb[-3:]
        self.right_fitc = self.right_fitc[-3:]
        self.right_fita.append(right_fit[0])
        self.right_fitb.append(right_fit[1])
        self.right_fitc.append(right_fit[2])
        self.avg_lft_fit[0]= np.average(self.left_fita)
        self.avg_lft_fit[1]= np.average(self.left_fitb)
        self.avg_lft_fit[2]= np.average(self.left_fitc)
        self.avg_rgt_fit[0]= np.average(self.right_fita)
        self.avg_rgt_fit[1]= np.average(self.right_fitb)
        self.avg_rgt_fit[2]= np.average(self.right_fitc)
        return(self.avg_lft_fit, self.avg_rgt_fit)
    def increment_bad_frames(self):
        self.bad_frames += 1
        self.linesdetected = False
        return(int(self.bad_frames))
    def set_frame(self, frame):
        self.frame=frame


def feed_the_beast(orig_imgBGR, data_store, convolve=False):
    orig_imgRGB = cv2.cvtColor(orig_imgBGR, cv2.COLOR_BGR2RGB)

    threshd_bin, truimgBGR, brdseyeBGR = line_pipeline(orig_imgBGR, convolve=convolve)

    data_store.add_new_image(threshd_bin)

    speed = estimate_speed(data_store)

    #find_center_line(threshd_bin)

    blank = np.zeros_like(threshd_bin, dtype=np.uint8)
    det = np.dstack((blank,threshd_bin,blank))*255

    truimgRGB = cv2.cvtColor(truimgBGR, cv2.COLOR_BGR2RGB)

    small_brdseyeBGR = cv2.resize(brdseyeBGR,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
    small_brdseyeRGB = cv2.cvtColor(small_brdseyeBGR, cv2.COLOR_BGR2RGB)

    small_det = cv2.resize(det, None, fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)

#    blank = np.zeros_like(small_det)
#    small_det = np.dstack((blank,small_det,blank))*255

    new_warp, line_fit, curve_radii, offcenter  = find_lines(data_store)

    truimgpluslaneRGB = cv2.addWeighted(truimgRGB, 1, new_warp, 0.3, 0)
    croptruimgpluslaneRGB = undistort_crop(truimgpluslaneRGB)

    curvature = (sum(data_store.curvatureL)+sum(data_store.curvatureR))/(len(data_store.curvatureR)+len(data_store.curvatureL)+0.01)

    small_line_fit = cv2.resize(line_fit,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)

    croptruimgpluslaneRGB[10:330,679:859,:] = small_brdseyeRGB
    croptruimgpluslaneRGB[10:330,879:1059,:] = small_det
    croptruimgpluslaneRGB[10:330,1079:1259,:] = small_line_fit
    cv2.putText(croptruimgpluslaneRGB, "Birdseye", (720,25), cv2.FONT_HERSHEY_SIMPLEX,0.75,(100,255,255),1,cv2.LINE_AA)
    cv2.putText(croptruimgpluslaneRGB, "Thresholds of", (905,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,255,255),1,cv2.LINE_AA)
    cv2.putText(croptruimgpluslaneRGB, "Sobel & Color Chans", (885,35), cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,255,255),1,cv2.LINE_AA)
    cv2.putText(croptruimgpluslaneRGB, "Line Fit", (1120,25), cv2.FONT_HERSHEY_SIMPLEX,0.75,(100,255,255),1,cv2.LINE_AA)
    cv2.putText(croptruimgpluslaneRGB, "Curve  Radius: {0:5,.0f} m".format(curvature), (40,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(croptruimgpluslaneRGB, "Left   A:{0:.6f} B:{1:.5f} C:{2:3.0f}".format(data_store.avg_lft_fit[0],
                                                                                      data_store.avg_lft_fit[1],
                                                                                      data_store.avg_lft_fit[2]
                                                                                     ), (40,150), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(croptruimgpluslaneRGB, "Right  A:{0:.6f} B:{1:.5f} C:{2:3.0f}".format(data_store.avg_rgt_fit[0],
                                                                                      data_store.avg_rgt_fit[1],
                                                                                      data_store.avg_rgt_fit[2]
                                                                                     ), (40,180), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(croptruimgpluslaneRGB, "Frame:{0:5d} Timestamp:{1:3.03f}".format(data_store.frame, (data_store.frame/data_store.fps)),
                                                                                    (40,210), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1,cv2.LINE_AA)


#    cv2.putText(croptruimgpluslaneRGB, "Left  Radius: {0:5,.0f} m".format(curve_radii[0]), (40,70), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
#    cv2.putText(croptruimgpluslaneRGB, "Right Radius: {0:5,.0f} m".format(curve_radii[1]), (40,110), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(croptruimgpluslaneRGB, "Off Lane Center: {0:2.2f} m".format(offcenter), (40,70), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(croptruimgpluslaneRGB, "Speed: {0:3.0f} km/hr".format(speed), (40,110), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)

    return croptruimgpluslaneRGB

def find_lines(data_store):
    binary_warped = np.copy(data_store.new_image)
    #flip the image so that polynomial coefficient 2 represents line location at the front of the car (not on the horizon)
    binary_warped = cv2.flip(binary_warped,flipCode=0)
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image
    '''
    The birdseye image is 720 wide by 1280 high
    the center of the camera and car is at 720/2 = 360
    lane lines are expected to be at about 260 and 460
            about 100 pixel offsets from center
    take a histogram of top 1/3 of image (since it's flipped) from x = 200 to 520
    to look for lane lines
    '''


    left_edge = 200
    right_edge = 520


    # Create an output image to draw on and  visualize the result
    out_img1 = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img1)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped[0:int(binary_warped.shape[0]*2/3),:].nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Set the width of the windows +/- margin
    margin = 40
    # Set minimum number of pixels found to recenter window
    minpix = 45
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    if data_store.linesdetected == False: # or (data_store.frame%10) == 0:

        histogram = np.sum(binary_warped[:binary_warped.shape[0]//3,left_edge:right_edge], axis=0)

        midpoint = np.int(histogram.shape[0]/2)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        leftx_base = np.argmax(histogram[:midpoint]) + left_edge
        #    print("left lane at {}".format(leftx_base))
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint + left_edge
        #    print("right lane at {}".format(rightx_base))

        # Choose the number of sliding windows
        nwindows = 10 # - need to make so that
        # Set height of windows
        #window_height = np.int(binary_warped.shape[0]/nwindows)
        #reduce height looking for lanes to get more confidant result
        window_height = np.int(binary_warped.shape[0]*(2/3)/nwindows)
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            #win_y_low = binary_warped.shape[0] - (window+1)*window_height
            #win_y_high = binary_warped.shape[0] - window*window_height
            win_y_low = window*window_height
            win_y_high = (window+1)*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img1,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img1,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        data_store.set_lines_detected(True)

    else: #search from last detectd  line
        left_fit = data_store.avg_lft_fit
        right_fit = data_store.avg_rgt_fit
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                           (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                           (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if (len(lefty)>0) & (len(leftx)>0) & (len(rightx)>0) & (len(righty)>0):

        old_left_fit = data_store.avg_lft_fit
        old_right_fit = data_store.avg_rgt_fit

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        #left_fit, right_fit = data_store.save_fit(left_fit,right_fit)
        # check if seems right at front of car (1260)
        #print("fnd lft_fita={0:3.5f}, lft_fitb={1:3.5f}, lft_fitc={2:3.5f}".format(left_fit[0],left_fit[1],left_fit[2]), end ='')
        #print("fnd rht_fita={0:3.5f}, rht_fitb={1:3.5f}, rht_fitc={2:3.5f}".format(right_fit[0],right_fit[1],right_fit[2]))


        if left_fit[2] > 220 and left_fit[2] < 300 and right_fit[2]-left_fit[2]>170 and right_fit[2]-left_fit[2]<230:
            #evrything looks ok at one end
            left_fit, right_fit = data_store.save_fit(left_fit,right_fit)
        elif left_fit[2] > 220 and left_fit[2] < 300 and (right_fit[2]-left_fit[2]<=170 or right_fit[2]-left_fit[2]>=230):
            right_fit[2] = left_fit[2] + 200
            right_fit[1] = left_fit[1]
            right_fit[0] = left_fit[0]
            left_fit, right_fit = data_store.save_fit(left_fit,right_fit)
        elif (left_fit[2] <= 220 or left_fit[2] >= 300) and (right_fit[2] > 420 and right_fit[2] < 500):
            left_fit[2] = right_fit[2] - 200
            left_fit[1] = right_fit[1]
            left_fit[1] = right_fit[1]
            left_fit, right_fit = data_store.save_fit(left_fit,right_fit)
        else:
            left_fit = old_left_fit
            right_fit = old_right_fit
            left_fit, right_fit = data_store.save_fit(left_fit,right_fit)
            data_store.increment_bad_frames()
        #print("Fix lft_fita={0:3.5f}, lft_fitb={1:3.5f}, lft_fitc={2:3.5f}".format(left_fit[0],left_fit[1],left_fit[2]), end='')
        #print("Fix rht_fita={0:3.5f}, rht_fitb={1:3.5f}, rht_fitc={2:3.5f}".format(right_fit[0],right_fit[1],right_fit[2]))

        # Generate x and y values for plotting

        ploty = np.linspace(0, int(binary_warped.shape[0]*2/3)-1, int(binary_warped.shape[0]*2/3) )


        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        #print()
        lane_center = (left_fitx[20] + right_fitx[20])/2
        #calculate displacement of center of lane from center of the image
        off_lane_center = (lane_center - 360)*3.7/200

        lane_width_check = np.average(right_fitx[0:100]-left_fitx[0:100])
        #print("Detected Lane width of {0:2.1f} pixels, which is off car center by {1:1.2f}m".format(lane_width_check, off_lane_center))
        #print("lane_width_check is {} pixels".format(lane_width_check))
        #if lane_width_check < 185 or lane_width_check > 230:
        #    #discard - use last good measurement
        #    left_fitx = data_store.left_fitx
        #    right_fitx = data_store.right_fitx
        #    frame = int(data_store.frame)
        #    bad_frames = int(data_store.increment_bad_frames())
        #    print("frame:{0:4d} rejected lane width {1:3.0f}- probably non parallel {2:2d} consecutive frames".format(frame, lane_width_check, bad_frames))
        #else:
        data_store.save_fitx(left_fitx,right_fitx)

        radii = find_radius( left_fitx, right_fitx, ploty )
        data_store.add_curvature(radii[0],radii[1])

        out_img1[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img1[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img1, 1, window_img, 0.5, 0)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        #cv2.minEnclosingCircle(pts_left)

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        color_warp = cv2.flip(color_warp, flipCode=0)
        result = cv2.flip(result, flipCode=0)

    else:
        result = color_warp # which will be blank
        radii = (0.,0.)
        off_lane_center = 0

    newwarp = birdseye_untransform(color_warp)
    return newwarp, result, radii, off_lane_center

'''
thoughts:
could try subtracting old image from new image and look for vertical size of positive blobs
or could convolve old with new and find maximum response (movement)  possibly try in smaller window to make faster
test1: try convolution /correlation - didn't achieve what I wanted
test2: Try least squares difference between section of new image slid over old image - works ok

'''
def estimate_speed(data_store, debug=False):
#    print("estimating speed")
    old_bin = np.copy(data_store.old_image)
    new_bin = np.copy(data_store.new_image)

    if np.ndim(old_bin) < 2:
        # no stored data
        print("no stored image - returning")
        speed = 0
        return speed
    else:
        #take a slice of road with lane lines on it in new binary image
        #then take a histogram of the pixels in it vertically, and flip it
        #so that the bottom of the image is now found at the start of the histogram.
        new_bin = np.flipud(np.sum(new_bin[1000:1200,200:520],axis=1))
        #take a taller slice of road with lane lines on it from last binary image
        # this should have the section of road see at the bottom of the new image
        # slightly higher up the image (and further along the histogram)
        # the distance for the match depends on the speed of the car and the fps of the video
        # but at 100 km/hr and 25 fps, we should expect about 25 pixels displacement
        # a window of 100 should cover other scenarios - this does limit resolution of speedo
        # to about 4 km/hr in this scenario
        old_bin = np.flipud(np.sum(old_bin[900:1200,200:520], axis=1))

        if debug == True:
            plt.plot(new_bin)
            plt.show()
            plt.plot(old_bin)
            plt.show()
        difference = np.array([])
        for i in range(1,100):
            old_bin_window = old_bin[i-1:199+i]
            ssd = ((new_bin - old_bin_window)**2).sum()
            difference = np.append(difference, ssd)
        displacement_pixels = ( np.argmin(difference))
        if debug == True:
            plt.plot(difference)
            plt.show()
            print("displacement is {0:3.1f} pixels.".format(displacement_pixels))

        data_store.add_displacement(displacement_pixels)
        frames_used = len(data_store.displacement)
        pixel_disp_in_muliple_frames = sum(data_store.displacement)
        meter_disp = pixel_disp_in_muliple_frames * (3.0/90)
        speed = meter_disp * (3600 / 1000) * (data_store.fps/frames_used)
        return speed

def find_radius( leftx, rightx, ploty):

    ym_per_pix = 3.0/90 # meters per pixel in y dimension
    xm_per_pix = 3.7/200 # meters per pixel in x dimension

    #y_eval = np.max(ploty)
    y_eval = 20

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    return (left_curverad, right_curverad)
