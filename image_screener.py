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
from operator import itemgetter

#%config InlineBackend.figure_format='svg'
# length of side of sliding windows used
boxdim1 = 80
boxdim2 = 120
boxdim3 = 160

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
# set up for boxes with 3 sets of dimensions, put each into separate plane to visualize effect
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    # separate heatmaps by box size to see what is hitting where
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        if box[1][1]-box[0][1] == boxdim1:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0],0] += 1
        elif box[1][1]-box[0][1] == boxdim2:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0],1] += 1
        elif box[1][1]-box[0][1] == boxdim3:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0],2] += 1
        else:
            print("missed a hit with box = {}".format(box))
    # Return updated heatmap
    return heatmap

#Function to set to zero all pixels below threshold
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmapf = heatmap.sum(axis=2)
    #print("heatmapf shape is {}".format(heatmapf.shape))
    heatmapf[heatmapf < threshold] = 0
    # Return thresholded map
    return heatmapf

def show_hit(heatmap):
    # only use binary for heathit
    heathitmap = np.zeros((heatmap.shape[0],heatmap.shape[1]),dtype=np.uint8)
    heathitmap[heatmap > 0] = 1
    # Return thresholded map
    return heathitmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    boxcolors = ((0,255,0),(0,0,255),(0,255,255),(255,0,0),(255,255,255),(255,255,255),(255,255,255),(255,255,255))
    bbox = []
    for car_number in range(1,labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

    # sort by right co-ordinate to try to keep car boxes having same colors (steady cars to right, oncoming cars to left)
    bbox = sorted(bbox, key=lambda x: x[1][0])
    colorindex = 0
    for boxnum in range(labels[1]-1,-1,-1):
        # Draw the box on the image
        cv2.rectangle(img, bbox[boxnum][0], bbox[boxnum][1], boxcolors[colorindex], 2)
        if colorindex > len(boxcolors) - 1:
            colorindex = len(boxcolors) - 1
        else:
            colorindex += 1
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
        self.heatstackvals = np.zeros(shape=(7, 720, 1280, 3), dtype=np.uint8) # was 20
        #one stack which has boolean for value above the threshold
        self.heatstack = np.zeros(shape=(7, 720, 1280), dtype=np.uint8) # was 20
        self.heatstackptr = 0
        self.prevstackptr = 0
        #overlapratioval = 0.5
    def addheat(self, heatmap, threshold):
        # store the raw heat values for each set of box sizes
        #print("addheat heatmap sum is {} with ptr = {}".format(heatmap.sum(),self.heatstackptr))

        self.heatstackvals[self.heatstackptr] = np.copy(heatmap)
        # apply the threshold
        heatmap = apply_threshold(heatmap, threshold)
        # equalize all values to unit value
        self.heatstack[self.heatstackptr] = show_hit(heatmap)
        self.prevstackptr = self.heatstackptr
        if self.heatstackptr+1 > 6: #19
            self.heatstackptr = 0
        else:
            self.heatstackptr += 1
        return np.sum(self.heatstack, axis=0).astype(np.uint8)
    def getheat(self):
        return np.sum(self.heatstack, axis=0).astype(np.uint8)
    def getheatval(self):
            return np.sum(self.heatstackvals, axis=0).astype(np.uint8)
    def getlastheatval(self):
            return np.copy(self.heatstackvals[self.prevstackptr])
    def set_overlap(self, overlapval):
        self.overlapratioval = overlapval

def find_cars(test_image, carfinder, heat_clip_threshold, history_bar=15, external_markup=False):

    box_dim80 = (80,80)
    test_strip80 = np.copy(test_image[400:540,80:1280,:]) #for 80x80
    origin_80 = (80,400)

    box_dim120 = (120,120)
    test_strip120 = np.copy(test_image[400:580,80:1280,:]) #for 120x120
    origin_120 = (80,400)

    box_dim160 = (160,160)
    test_strip160 = np.copy(test_image[400:640,160:1280,:]) #for 160x160
    origin_160 = (160,400)

    #test_strip200 = np.copy(test_image[350:650,380:1280,:]) #for 200x200

    t=time.time()

    features_80, strip1_images, window_locs_80 = process_windows(test_strip80, windowshape=box_dim80,
                            hogwindowsize=carfinder.hogsizeval, colorwindowsize=carfinder.spatialsizeval,
                            overlapratio=carfinder.overlapratioval, orient=carfinder.orientval,
                            pix_per_cell=carfinder.pixpercellval, cell_per_block=carfinder.cellperblockval,
                            nbins=carfinder.histbinsval, bins_range=carfinder.histrangeval,
                            origin=origin_80)

    #features_120, strip1_images, window_locs_120 = process_windows(test_strip120, windowshape=box_dim120,
    #                        hogwindowsize=carfinder.hogsizeval, colorwindowsize=carfinder.spatialsizeval,
    #                        overlapratio=carfinder.overlapratioval, orient=carfinder.orientval,
    #                        pix_per_cell=carfinder.pixpercellval, cell_per_block=carfinder.cellperblockval,
    #                        nbins=carfinder.histbinsval, bins_range=carfinder.histrangeval,
    #                        origin=origin_120)

    #features_160, strip1_images, window_locs_160 = process_windows(test_strip160, windowshape=box_dim160,
    #                        hogwindowsize=carfinder.hogsizeval, colorwindowsize=carfinder.spatialsizeval,
    #                        overlapratio=carfinder.overlapratioval, orient=carfinder.orientval,
    #                        pix_per_cell=carfinder.pixpercellval, cell_per_block=carfinder.cellperblockval,
    #                        nbins=carfinder.histbinsval, bins_range=carfinder.histrangeval,
    #                        origin=origin_160)

    #features_200, strip1_images, window_locs_200 = process_windows(test_strip200, windowshape=(200,200),
    #                        hogwindowsize=carfinder.hogsizeval, colorwindowsize=carfinder.spatialsizeval,
    #                        #overlapratio=carfinder.overlapratioval, orient=carfinder.orientval,
    #                        pix_per_cell=carfinder.pixpercellval, cell_per_block=carfinder.cellperblockval,
    #                        nbins=carfinder.histbinsval, bins_range=carfinder.histrangeval,
    #                        origin=(480,350))

    features = features_80
    #features.extend(features_120)
    #features.extend(features_160)

    window_locs = window_locs_80
    #window_locs.extend(window_locs_120)
    #window_locs.extend(window_locs_160)

    features = np.vstack((features)).astype(np.float64)
    scaled_features = carfinder.X_scaler.transform(features)
    choices = carfinder.svc.predict(scaled_features)
    t2 = time.time()
    #print('{} Seconds to calculate features for {} images'.format(round(t2-t, 6), len(choices)))
    #print('SVC predicts: {}'.format(choices))

    boxes = []
    #store all the patches that svc matched on in boxes[]
    for i in choices.nonzero()[0]:
    #for i in range(len(window_locs)): #use this line to show all boxes
        boxes.append(window_locs[i])

    #dynamic_heat_threshold = len(boxes)//5
    #if dynamic_heat_threshold > heat_clip_threshold:
    #dynamic_heat_threshold = heat_clip_threshold

    # generate an RGB image to display
    test_image_rgb = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
    test_boxes = draw_boxes(test_image_rgb,boxes)

    blankheat = np.zeros((test_image.shape[0], test_image.shape[1], 3), dtype=np.uint8)
    heat = add_heat(blankheat, boxes)
    heatmap = carfinder.addheat(heat, heat_clip_threshold)
    heatmap[heatmap<history_bar]=0 #by default only show if more than 15 matches in history

    #keep everything in range
    heatmap = np.clip(heatmap, 0, 255)
    if external_markup == False:
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(test_image_rgb), labels)
        return draw_img, boxes, heatmap, heat, test_boxes
    else:
        return

'''
%matplotlib inline

road_images = glob.glob('./test_images/*.jpg')
i=1
for frame in road_images:
    car_values = car_finder()
    test_image = cv2.imread(frame)
    draw_img, boxes, histheat, heat, test_boxes = find_cars(test_image, car_values, 4, history_bar=1)

    fig = plt.figure(figsize=(18,6))
    plt.imshow(test_boxes)
    #plt.title('80 pixel Sliding Window Search')
    plt.title('Window Search Hits')
    fig.tight_layout()
    #plt.savefig("./output_images/figure8.png")

    fig = plt.figure(figsize=(18,6))
    plt.imshow(draw_img)
    plt.title('Cars Located')
    fig.tight_layout()
    imagefilename="./output_images/figure10_"+str(i)+".png"
    plt.savefig(imagefilename)

    heat = car_values.getlastheatval()
    heat = heat * 40
    fig = plt.figure(figsize=(18,6))
    plt.imshow(heat,cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    #plt.savefig("./output_images/figure9.png")

    fig = plt.figure(figsize=(18,6))
    plt.imshow(histheat,cmap='hot')
    plt.title('Binary Heat Map')
    fig.tight_layout()

    i += 1



    #plt.imshow(heat,cmap='hot')
'''
