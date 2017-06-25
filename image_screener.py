import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from feature_extraction import *
from MultiWindow import *
import glob
import random
import time
import pickle

#%config InlineBackend.figure_format='svg'

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
# Function to increment pixel values within boxes
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap

#Function to set to zero all pixels below threshold
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap

def show_hit(heatmap):
    # only use binary for heat
    heatmap[heatmap > 0] = 1
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    boxcolors = ((0,255,0),(0,0,255),(0,255,255),(255,0,0),(255,255,255),(255,255,255),(255,255,255),(255,255,255))
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], boxcolors[1], 2)
    # Return the image
    return img


class car_finder():
    def __init__(self):
        dist_pickle = pickle.load( open(".trained_svc2.p", "rb" ) )
        self.svc = dist_pickle["svc"]
        self.X_scaler = dist_pickle["scaler"]
        self.orientval = dist_pickle["orient"]
        self.pixpercellval = dist_pickle["pix_per_cell"]
        self.cellperblockval = dist_pickle["cell_per_block"]
        self.spatialsizeval = dist_pickle["spatial_size"]
        self.histbinsval = dist_pickle["hist_bins"]
        self.histrangeval = dist_pickle["hist_range"]
        self.hogchannelval = dist_pickle["hog_channel"]
        self.hogsizeval = dist_pickle["HOG_size"]
        self.cspaceval = dist_pickle["colorspace"]
        self.overlapratioval = dist_pickle["overlapratio"]
        #one stack which has heat values above the threshold
        self.heatstackvals = np.zeros(shape=(20, 720, 1280), dtype=np.uint8)
        #one stack which has boolean for value above the threshold
        self.heatstack = np.zeros(shape=(20, 720, 1280), dtype=np.uint8)
        self.heatstackptr = 0
        #overlapratioval = 0.5
    def addheat(self, heatmap):
        self.heatstackvals[self.heatstackptr] = heatmap
        self.heatstack[self.heatstackptr] = show_hit(heatmap)
        if self.heatstackptr+1 > 19:
            self.heatstackptr = 0
        else:
            self.heatstackptr += 1
        return np.sum(self.heatstack, axis=0).astype(np.uint8)
    def getheat(self):
        return np.sum(self.heatstack, axis=0).astype(np.uint8)
    def getheatval(self):
            return np.sum(self.heatstackvals, axis=0).astype(np.uint8)
    def getlastheatval(self):
            return self.heatstackvals[self.heatstackptr]
    def set_overlap(self, overlapval):
        self.overlapratioval = overlapval

def find_cars(test_image, carfinder, heat_clip_threshold, history_bar=15):

    #test_strip40 = np.copy(test_image[400:480,320:1280,:]) #for 40*40
#    test_strip80 = np.copy(test_image[400:520,320:1280,:]) #for 80x80
#    test_strip120 = np.copy(test_image[400:580,320:1280,:]) #for 120x120
#    test_strip160 = np.copy(test_image[400:600,320:1280,:]) #for 160x160
    test_strip80 = np.copy(test_image[400:520,80:1280,:]) #for 80x80
    origin_80 = (80,400)
    test_strip120 = np.copy(test_image[400:580,80:1280,:]) #for 120x120
    origin_120 = (80,400)
    test_strip160 = np.copy(test_image[400:640,160:1280,:]) #for 160x160
    origin_160 = (160,400)
    #test_strip200 = np.copy(test_image[350:650,380:1280,:]) #for 200x200

    t=time.time()

    features_80, strip1_images, window_locs_80 = process_windows(test_strip80, windowshape=(80,80),
                            hogwindowsize=carfinder.hogsizeval, colorwindowsize=carfinder.spatialsizeval,
                            overlapratio=carfinder.overlapratioval, orient=carfinder.orientval,
                            pix_per_cell=carfinder.pixpercellval, cell_per_block=carfinder.cellperblockval,
                            nbins=carfinder.histbinsval, bins_range=carfinder.histrangeval,
                            origin=origin_80)

    features_120, strip1_images, window_locs_120 = process_windows(test_strip120, windowshape=(120,120),
                            hogwindowsize=carfinder.hogsizeval, colorwindowsize=carfinder.spatialsizeval,
                            overlapratio=carfinder.overlapratioval, orient=carfinder.orientval,
                            pix_per_cell=carfinder.pixpercellval, cell_per_block=carfinder.cellperblockval,
                            nbins=carfinder.histbinsval, bins_range=carfinder.histrangeval,
                            origin=origin_120)

    features_160, strip1_images, window_locs_160 = process_windows(test_strip160, windowshape=(160,160),
                            hogwindowsize=carfinder.hogsizeval, colorwindowsize=carfinder.spatialsizeval,
                            overlapratio=carfinder.overlapratioval, orient=carfinder.orientval,
                            pix_per_cell=carfinder.pixpercellval, cell_per_block=carfinder.cellperblockval,
                            nbins=carfinder.histbinsval, bins_range=carfinder.histrangeval,
                            origin=origin_160)

    #features_200, strip1_images, window_locs_200 = process_windows(test_strip200, windowshape=(200,200),
    #                        hogwindowsize=carfinder.hogsizeval, colorwindowsize=carfinder.spatialsizeval,
    #                        #overlapratio=carfinder.overlapratioval, orient=carfinder.orientval,
    #                        pix_per_cell=carfinder.pixpercellval, cell_per_block=carfinder.cellperblockval,
    #                        nbins=carfinder.histbinsval, bins_range=carfinder.histrangeval,
    #                        origin=(480,350))

    features = features_80
    features.extend(features_120)
    features.extend(features_160)

    window_locs = window_locs_80
    window_locs.extend(window_locs_120)
    window_locs.extend(window_locs_160)

    features = np.vstack((features)).astype(np.float64)
    scaled_features = carfinder.X_scaler.transform(features)
    choices = carfinder.svc.predict(scaled_features)
    t2 = time.time()
    #print('{} Seconds to calculate features for {} images'.format(round(t2-t, 6), len(choices)))
    #print('SVC predicts: {}'.format(choices))

    boxes = []
    #store all the patches that svc matched on in boxes[]
    for i in choices.nonzero()[0]:
        boxes.append(window_locs[i])

    #dynamic_heat_threshold = len(boxes)//5
    #if dynamic_heat_threshold > heat_clip_threshold:
    #dynamic_heat_threshold = heat_clip_threshold

    # generate an RGB image to display
    test_image_rgb = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
    test_boxes = draw_boxes(test_image_rgb,boxes)

    blankheat = np.zeros_like(test_image[:,:,0]).astype(np.uint8)

    heat = add_heat(blankheat, boxes)

    #save heat - maybe can use in video across multiple frames
    heatmap = apply_threshold(heat, heat_clip_threshold)
    #heatmap = apply_threshold(heat, dynamic_heat_threshold)
    historyheatmap = carfinder.addheat(heatmap)
    heatmap = apply_threshold(historyheatmap, history_bar) #by default only show if more than 15 matches in history
    #keep everything in range
    heatmap = np.clip(heatmap, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(test_image_rgb), labels)

    return draw_img, boxes, heatmap, heat

#%matplotlib inline

road_images = glob.glob('./test_images/*.jpg')
for frame in road_images:
    car_values = car_finder()
    test_image = cv2.imread(frame)
    draw_img, boxes, histheat, heat = find_cars(test_image, car_values, 6, history_bar=1)
    fig = plt.figure(figsize=(18,6))
    plt.imshow(draw_img)
    fig = plt.figure(figsize=(18,6))

    plt.imshow(histheat,cmap='hot')
    heat = car_values.getheatval()
    heat = heat * 10
    fig = plt.figure(figsize=(18,6))
    plt.imshow(heat,cmap='hot')
