import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from feature_extraction import *
import glob
import random
import time
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC


cars = glob.glob('./vehicles/GTI_Far/*.png')
cars.extend( glob.glob('./vehicles/GTI_Left/*.png'))
cars.extend( glob.glob('./vehicles/GTI_Right/*.png'))
cars.extend( glob.glob('./vehicles/GTI_MiddleClose/*.png'))
cars.extend( glob.glob('./vehicles/KITTI_forTraining/*.png'))

notcars = glob.glob('./non-vehicles/GTI/*.png')
notcars.extend( glob.glob('./non-vehicles/curated/*.png')) # files from test set and cut from video
print("{} cars and {} notcars in input training set".format(len(cars), len(notcars)) )

#equalize the sets of cars and not cars
if (len(cars) > len(notcars)):
    cars = random.sample(cars,len(notcars))
else:
    notcars = random.sample(notcars,len(cars))

print("Adjusted to {} cars and {} notcars in training set".format(len(cars), len(notcars)) )

train_len = len(cars)*2

cspaceval='YCrCb'
spatialsizeval=(8, 8)
#spatialsizeval=(6, 6)
histbinsval=16
histrangeval=(0, 256)
overlapratioval = 0.25
#hog parameters
hogcheckval = True
#hogsizeval = (20,20)
hogsizeval = (16,16)
hogchannelval = 'ALL'
#bin orientations of gradient
orientval=9

pixpercellval=4

# will result in 3*3 for hog results in an image

# think next is right - should work out to 2
cellperblockval= 2 #np.int(hogsizeval[0]/pixpercellval*overlapratioval)


testcars = glob.glob('./vehicles/KITTI_extracted/*.png')

testnotcars = glob.glob('./non-vehicles/Extras/*.png')
print("{} cars and {} notcars in test set".format(len(testcars), len(testnotcars)) )
#compensate for size of one set being largerthan the other
if (len(testcars) > len(testnotcars)):
    testcars = random.sample(testcars,len(testnotcars))
else:
    testnotcars = random.sample(testnotcars,len(testcars))

print("Adjusted to {} cars and {} notcars in test set".format(len(testcars), len(testnotcars)) )

car_features = extract_features(cars,
                        cspace=cspaceval, spatial_size=spatialsizeval, hist_bins=histbinsval,
                        hist_range=histrangeval, hogcheck=hogcheckval, hog_size=hogsizeval,
                        hog_channel=hogchannelval, pix_per_cell=pixpercellval,
                        orient=orientval, cell_per_block=cellperblockval)
notcar_features = extract_features(notcars,
                        cspace=cspaceval, spatial_size=spatialsizeval, hist_bins=histbinsval,
                        hist_range=histrangeval, hogcheck=hogcheckval, hog_size=hogsizeval,
                        hog_channel=hogchannelval, pix_per_cell=pixpercellval,
                        orient=orientval, cell_per_block=cellperblockval)

testcar_features = extract_features(testcars,
                        cspace=cspaceval, spatial_size=spatialsizeval, hist_bins=histbinsval,
                        hist_range=histrangeval, hogcheck=hogcheckval, hog_size=hogsizeval,
                        hog_channel=hogchannelval, pix_per_cell=pixpercellval,
                        orient=orientval, cell_per_block=cellperblockval)

testnotcar_features = extract_features(testnotcars,
                        cspace=cspaceval, spatial_size=spatialsizeval, hist_bins=histbinsval,
                        hist_range=histrangeval, hogcheck=hogcheckval, hog_size=hogsizeval,
                        hog_channel=hogchannelval, pix_per_cell=pixpercellval,
                        orient=orientval, cell_per_block=cellperblockval)




if len(car_features) > 0:
    # Create an array stack of feature vectors
    X_train = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_testset = np.vstack((testcar_features, testnotcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    scaled_X_train = X_scaler.transform(X_train)
    scaled_X_test = X_scaler.transform(X_testset)

y_train = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

y_testset = np.hstack((np.ones(len(testcar_features)), np.zeros(len(testnotcar_features))))
rand_state = np.random.randint(0, 100)
X, y = shuffle(scaled_X_train, y_train, random_state=rand_state)
print("Run on test set of {} samples".format(len(X)//5))
rand_state = np.random.randint(0, 100)
X_test, y_test = shuffle(scaled_X_test, y_testset, random_state=rand_state, n_samples=int(len(X)//5))
print("shuffled {} test samples".format(len(X_test)))
rand_state = np.random.randint(0, 100)

tuned_parameters = [{'kernel':['linear'], 'C':[0.1, 1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
