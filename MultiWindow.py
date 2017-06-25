import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import random
import time
from feature_extraction import *

#slide windows
# process an image to identify cars
# input image is expected in BGR format
# windowshape is size of patches to be evaluated in dimensions of original image
# hogwindowsize is dimension in pixels that patches should be resized to for HOG comparison
# colorwindowsize is dimension in pixels that patches should be resized to for color analysis
def process_windows(image, windowshape, hogwindowsize, colorwindowsize, overlapratio,
                    orient, pix_per_cell, cell_per_block, nbins,
                    bins_range, origin=(0,0)):

    #determine dimensions of input image
    oldwidth,oldheight = image.shape[1::-1]
    # how many windows wide is the strip to be evaluated (no overlap)
    windowswide = oldwidth/windowshape[0]
    # how many windows high is the strip to be evaluated
    windowshigh = oldheight/windowshape[1]

    # how many windows will we evaluate in each horizontal strip
    # overalpratio must be multiple of 1/4 since 8x8 in color patches, and 4 blocks when crossing HOG features.
    windowsperstrip = np.int((windowswide-1)/overlapratio + 1)
    windowstrips = np.int((windowshigh-1)/overlapratio + 1)

    #calculate size in pixels that image is to resized to for HOG and color analysis
    newhogWidth = np.int(windowswide*hogwindowsize[0])
    newhogHeight = np.int(windowshigh*hogwindowsize[1])
    newcolorWidth = np.int(windowswide*colorwindowsize[0])
    newcolorHeight = np.int(windowshigh*colorwindowsize[1])

    imagewindows = []

    #resize for Hog analysis which uses a larger image window.
    image_r = cv2.resize(image,(newhogWidth,newhogHeight), interpolation=cv2.INTER_AREA)
    #should set up to use alternate colorspaces here
    image_hog = cv2.cvtColor(image_r, cv2.COLOR_BGR2YCrCb)
    #resize for color analysis which uses a smaller
    image_color = cv2.resize(image_hog,(newcolorWidth,newcolorHeight), interpolation=cv2.INTER_AREA)
    #print("color shape is {}".format(image_color.shape))
    #run HOG analysis on whole resized HOG image
    image_hog_feats = []
    for channel in range(image_hog.shape[2]):
        image_hog_feats.append(get_hog_features(image_hog[:,:,channel], orient,
                pix_per_cell, cell_per_block, vis=False, feature_vec=False, transform_sq=True))
    #print("hog features shape is {}".format(image_hog_feats[0].shape))
    #print("hog features dtype is {}".format(image_hog_feats[0].dtype))

    nblocks_per_window = (hogwindowsize[0] // pix_per_cell) - cell_per_block + 1
    blocks_per_step = np.int(overlapratio*(nblocks_per_window+1))  # Instead of overlap, define how many cells to step
    pix_per_step = np.int(overlapratio*colorwindowsize[0]) #this needs to be fixed for other overlap ratios than 0.5

    #print("nxsteps is {} pix_per_step is {}".format(windowsperstrip,pix_per_step))
    #print("nblocks_per_window is {} blocks_per_step is {}".format(nblocks_per_window,blocks_per_step))

    features = []
    strip_hog = []
    strip_images = []
    for yb in range(windowstrips):
        for xb in range(windowsperstrip):
            ypos = yb*blocks_per_step
            xpos = xb*blocks_per_step
            # Extract HOG for this patch
            hog_feats =[]
            if ypos + nblocks_per_window > image_hog_feats[0].shape[0]:
                print("ypos out of bounds is {} max is {}".format(ypos,image_hog_feats[0].shape[0]))
            if xpos + nblocks_per_window > image_hog_feats[0].shape[1]:
                print("xpos out of bounds is {} max is {}".format(xpos,image_hog_feats[0].shape[1]))
            hog_feats.append(np.ravel(image_hog_feats[0][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]))
            hog_feats.append(np.ravel(image_hog_feats[1][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]))
            hog_feats.append(np.ravel(image_hog_feats[2][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window]))
            hog_feats = np.ravel(hog_feats)
            #print("{}: hog_feats len {}".format((xb,yb),len(hog_feats)))
            #now grab image of this patch too
            ypospix = yb*pix_per_step
            xpospix = xb*pix_per_step
            image_patch = image_color[ypospix:ypospix+colorwindowsize[1],xpospix:xpospix+colorwindowsize[0],:]
            strip_images.append(image_patch)
            col_spatial_features = bin_spatial(image_patch, size=colorwindowsize)
            col_hist_features = color_hist(image_patch, nbins=nbins, bins_range=bins_range, size=colorwindowsize)
            #print("{}: col_hist_features: {}".format((xb,yb), col_hist_features.shape))
            #print("{}: col_spatial_features: {}".format((xb,yb), col_spatial_features.shape))
            #print("{}: hog_feats: {}".format((xb,yb),hog_feats.shape))
            feats = np.concatenate((col_hist_features, col_spatial_features,hog_feats), axis=0)
            features.append(feats)
            #store original image window co-ordinates for plotting
            xtop = np.int(origin[0]+(xpospix*(oldwidth/newcolorWidth)))
            yleft = np.int(origin[1]+(ypospix*(oldheight/newcolorHeight)))
            #print("window - ({}) to ({})".format((xtop,yleft),(xtop+windowshape[0],yleft+windowshape[1])))
            imagewindows.append([(xtop,yleft),(xtop+windowshape[0],yleft+windowshape[1])])

    return features, strip_images, imagewindows
