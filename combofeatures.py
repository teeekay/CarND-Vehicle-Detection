import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from feature_extraction import *
%matplotlib inline
%config InlineBackend.figure_format='svg'
import glob
import random
import time
import pickle

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

    car_ind = np.random.randint(0, len(cars))
    fig = plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('Training Car Image')
    plt.subplot(222)
    plt.imshow(mpimg.imread(notcars[car_ind]))
    plt.title('Training Not-Car Image')
    plt.subplot(223)
    plt.imshow(mpimg.imread(testcars[car_ind]))
    plt.title('Test Car Image')
    plt.subplot(224)
    plt.imshow(mpimg.imread(testnotcars[car_ind]))
    plt.title('Test Not-Car Image')
    fig.tight_layout()
    #plt.savefig("./examples/figure2.png")

    car_ind = np.random.randint(0, len(cars))
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X_train[car_ind])
    plt.title('Raw Color and HOG Features')
    plt.subplot(133)
    plt.plot(scaled_X_train[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    #plt.savefig("./examples/figure3.png")

    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(notcars[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X_train[car_ind+len(cars)])
    plt.title('Raw Color and HOG Features')
    plt.subplot(133)
    plt.plot(scaled_X_train[car_ind+len(cars)])
    plt.title('Normalized Features')
    fig.tight_layout()
    #plt.savefig("./examples/figure4.png")
else:
    print('Your function only returns empty feature vectors...')

y_train = np.hstack((np.ones(len(car_features)),
              np.zeros(len(notcar_features))))

y_testset = np.hstack((np.ones(len(testcar_features)),
              np.zeros(len(testnotcar_features))))
from sklearn.utils import shuffle

rand_state = np.random.randint(0, 100)
X, y = shuffle(scaled_X_train, y_train, random_state=rand_state)
print("Run on test set of {} samples".format(len(X)//5))
rand_state = np.random.randint(0, 100)
X_test, y_test = shuffle(scaled_X_test, y_testset, random_state=rand_state, n_samples=int(len(X)//5))
print("shuffled {} test samples".format(len(X_test)))
rand_state = np.random.randint(0, 100)

from sklearn.svm import LinearSVC
# Use a linear SVC (support vector classifier)


svc = LinearSVC(C=10)
# Train the SVC
t=time.time()
svc.fit(X, y)
t2 = time.time()
print('{} Seconds to train SVC...'.format(round(t2-t, 2)))

print('Test Accuracy of SVC = {}%'.format((svc.score(X_test, y_test)*100)))

n_predict = 25
X_test, y_test = shuffle(scaled_X_test, y_testset, random_state=rand_state, n_samples=n_predict)
t=time.time()
print('My SVC predicts:\n{}'.format( svc.predict(X_test[0:n_predict])))
print('For these {} labels:\n{}'.format(n_predict, y_test[0:n_predict]))
t2 = time.time()
print('{} Seconds to predict {} labels with SVC'.format(round(t2-t, 6), n_predict))


#pickle up data needed to use this svc
out_set = {}
out_set["svc"] = svc
out_set["scaler"] = X_scaler
out_set["orient"] = orientval
out_set["pix_per_cell"] = pixpercellval
out_set["cell_per_block"] = cellperblockval
out_set["hog_channel"] = hogchannelval
out_set["HOG_size"] = hogsizeval
out_set["spatial_size"] = spatialsizeval
out_set["hist_bins"] = histbinsval
out_set["hist_range"] = histrangeval
out_set["colorspace"] = cspaceval
out_set["overlapratio"] = overlapratioval
with open('.trained_svc2.p', 'wb') as f:
    pickle.dump(out_set,f)
    f.close()
print("Processed SVC set pickled up")

'''
#code to print out HOG examples
image = cv2.imread(testcars[np.random.randint(0, len(testcars))])
image_y = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
image_r = cv2.resize(image_y,(16,16),interpolation=cv2.INTER_AREA)
features, hog_image = get_hog_features(image_r[:,:,0], 9, 4, 2, vis=True, feature_vec=False, transform_sq=True)
print(features.shape)
fig = plt.figure(figsize=(12,4))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(image)
plt.subplot(122)
plt.imshow(hog_image, cmap='gray')
plt.title('HOG Visualization')
fig.tight_layout()
plt.savefig("./examples/figure5.png")
'''
acc = (92.79055,93.5363,93.7228,93.225,93.22560)
c = (-1,0,1,2,3)
fig = plt.figure(figsize=(5,4))
plt.title('Log10 C versus test set Accuracy')
plt.scatter(c,acc)
fig.tight_layout()
plt.savefig("./output_images/figure6.png")
