import numpy as np
import cv2
from skimage.feature import hog

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True, transform_sq=False):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                    cells_per_block=(cell_per_block,cell_per_block), transform_sqrt=transform_sq,
                    visualise=True, feature_vector=feature_vec, block_norm='L2-Hys')
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                    cells_per_block=(cell_per_block,cell_per_block),  transform_sqrt=transform_sq,
                    visualise=False, feature_vector=feature_vec, block_norm='L2-Hys')
        return features


def bin_spatial(img, size=(32, 32)):
    # Convert image to new color space (if specified)
    # to create the feature vector
    if img.shape != size:
        image = cv2.resize(img,size, interpolation=cv2.INTER_AREA)
    else:
        image = np.copy(img)

    features = image.astype(np.float64).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256), size=(32, 32)):

    if img.shape != size:
        image = cv2.resize(img,size)
    else:
        image = np.copy(img)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(image[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(image[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(image[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0])).astype(np.float64)
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_colorchange(cspace):
    if cspace == 'HSV':
        colorchange = cv2.COLOR_BGR2HSV
    elif cspace == 'LUV':
        colorchange = cv2.COLOR_BGR2LUV
    elif cspace == 'HLS':
        colorchange =cv2.COLOR_BGR2HLS
    elif cspace == 'YUV':
        colorchange = cv2.COLOR_BGR2YUV
    elif cspace == 'YCrCb':
        colorchange = cv2.COLOR_BGR2YCrCb
    else:
        print("colorspace of {} not recognized, using BGR in extract_features.".format(cspace))
        colorchange = 0
    return colorchange


def extract_features(img_set, loadfromnames=True, cspace='BGR', spatial_size=(8, 8), hist_bins=32, hist_range=(0, 256),
                     hogcheck=True, hog_size=(16,16), hog_channel=0, orient=9, pix_per_cell=8, cell_per_block=2, pre_calced_hog=None):

    colorchange = 0
    shapeprinted = False
    if cspace != 'BGR':
        colorchange = get_colorchange(cspace)

    feature_list = []
    for i in range(len(img_set)):
        if loadfromnames == True:
            #img_set is a list of filenames
            image = cv2.imread(img_set[i])
        else:
            #img_set is a list of images in BGR format
            image = img_set[i]

        if colorchange != 0:
            image = cv2.cvtColor(image, colorchange)
        #resize if required
        if image.shape[1::-1] != spatial_size:
            image_r = cv2.resize(image, spatial_size,interpolation=cv2.INTER_AREA)
        else:
            image_r = image

        col_spatial_features = bin_spatial(image_r, size=spatial_size)
        col_hist_features = color_hist(image_r, nbins=hist_bins, bins_range=hist_range, size=spatial_size)

        feats = np.concatenate((col_hist_features, col_spatial_features), axis=0)

        if hogcheck == True:
            if pre_calced_hog == None:
                if image_r.shape[1::-1] == hog_size:
                    image_h = image_r
                elif image.shape[1::-1] == hog_size:
                    image_h = image
                else:
                    image_h = cv2.resize(image,hog_size,interpolation=cv2.INTER_AREA)

                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(image.shape[2]):
                        hog_features.append(get_hog_features(image_h[:,:,channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True, transform_sq=True))
                        #if shapeprinted == False:
                        #    print("Hog shape is {}".format(hog_features[channel].shape))
                        #    shapeprinted = True
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = get_hog_features(image_h[:,:,hog_channel], orient, pix_per_cell,
                                                    cell_per_block, vis=False, feature_vec=True, transform_sq=True)
            else:
                #(previously stored hog feature array passed in to function)
                hog_features = pre_calced_hog[i]

            feats = np.concatenate((feats, hog_features), axis=0)

        feature_list.append(feats)

    return feature_list
