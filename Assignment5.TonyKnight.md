## **Vehicle Detection Project**

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/videograb00002.png?raw=true"  width=1000>

<i><u>Figure 1 Snapshot from Vehicle Detection Video</u></i>

---


### Writeup by Tony Knight - 2017/06/01 


#### Training Set

I decided to use the GTI data set as the basis for training the Support Vector Machine Classifier.  However, this data set was slightly imbalanced, with more images which were "non-vehicles" than images of vehicles.  I supplemented the car dataset with 1200 images of cars from the KITTI dataset, producing a total of 4023 images of cars and 4047 images of non-cars in the dataset.  I also added a few samples of "curated" non-car images taken from the project video in order to attempt to train the classifier away from some negative hits.  

For the test dataset I used images of cars from the KITTI dataset (excluding those used in the training set) and non-car images from the extras dataset (taken from the video).  Figure 2 shows images of "cars" and "non-cars" used in the training and test sets.

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure2.png?raw=true"  width=700>

<i><u>Figure 2: Images of Cars and Non-cars used in training and Test sets</u></i>

---

### Training 

I decided to use a combination of the color and histogram of gradients (HOG) features to characterize each image patch.  I wanted to make the size of the patches as small as possible in  order to speed up processing during image analysis.  I decided that I would resize the image patches to 16x16 pixels to extract the HOG, and 8x8 pixels to extract color information.  Based on initial tests with the test images I decided to use the YCrCb color space, and to use all 3 of the color channels in the HOG analysis.


### Histogram of Oriented Gradients (HOG)

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure5.png?raw=true"  width=700>

<i><u>Figure 3: Image of Car and HOG visualization produced from Y color channel of image in YCrCB colorspace</u></i>

---

The code to run the HOG analysis is contained in function `extract_features()` and in function `get_hog_features()` in [feature_extraction.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/feature_extraction.py) on lines 94 through 117, and lines 5 to 16 respectively.  The scikit implementation of `hog()` was called from these functions to calculate the HOG features


I tried various combinations of parameters and ended up using 4 pixels per side on cells, and using 2x2 cells in a block.  This resulted in a total of 3x3 histograms of gradients from each image.  Part of the reason for selecting this size, was it facilitated implementation of a sliding window with overlap ratios of 0.25, or 0.5 when only calculating HOGs once on the whole image (as 4 histograms are passed over (3 in the patch and one on the boundary of the patch) when sliding a window across the width of a patch in one direction).

I chose to use 9 bins for the orientations of the histograms as it produced good results and using more bins did not improve results.

The selection of these parameters resulted in 3x3x2x2x9=324 HOG features for each  of the 3 color channels on the image patch (972 total). 

I set transform_sqrt to True, even though this slightly increased the hog calculation time, as it appeared to produce better results when I tested the implementation on the video.    

#### Color features

Using a patch size of 8x8 pixels, the color distribution for each of the Y, Cr and Cb color channels was split between 16 histogram bins in function `color_hist()` in [feature_extraction.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/feature_extraction.py) (lines 32 to 45)  resulting in 48 features.  The value of each pixel in each color channel was also used as a feature resulting in 3x8x8 = 192 features.

#### Scaler

The values of the 3 types of features were normalized using Scikits StandardScaler which was fit to the training set on line 99 of [combofeatures.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/combofeatures.py).  Figure 4 and 5 show images of cars and non-cars in the training set, and the resulting plots of features before and after normalization. 

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure3.png?raw=true"  width=700>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure4.png?raw=true"  width=700>


<i><u>Figures 4 and 5: Images of Car and non Car and resulting raw and scaled feature sets</u></i>

---


#### Training 

A linear SVM classifier was trained in [combofeatures.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/combofeatures.py) between lines 150 and 175. I adjusted the value of C between 0.1 and 1000 by orders of magnitude, but found there was little change in accuracy results on the test set.  The accuracy rates ranged between a low of 92.7 when C was 0.1 and a high of 93.7 when C was set at 10, and back down to 93.1 at C=1000 (The results are shown in figure 6)

The error rate generally was due to false positives for detection of cars on images that weren't cars.  Almost all car images were identified as cars.

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure6.png?raw=true"  width=400>

<i><u>Figure 6: Plot of Accuracy results as log10 C is varied</u></i>

---


### Sliding Window Search

I initially set up a sliding image search with 5 sizes of windows (edge lengths 40 pixels, 80 pixels, 120 pixels and 200 pixels).  After experimentation, and specifically after visualizing the response when using each size window, I realized that I was able to obtain the best results when using 80 pixel windows alone.  The smaller 40 pixel wide windows did not detect smaller cars on the horizon, but took a large amount of processing time, so I also dropped them.   I found that the use of windows larger than 80 pixels width added more false positives, but did not really add any better detection of cars that were not picked up by the 80 pixel windows.  Furthermore, I limit false positives in the videostream by using a buffer of heat images, and thresholding pixels that have had heat for a specified number of frames.  However, with the larger windows, I found that this caused larger areas to remain hot for long periods and preventing this required the use of a large history threshold (up to 15 frames) which caused a significant delay in displaying valid hits.  Removal of the larger window sizes enabled the history threshold to be reduced down to 5 frames, enabling much faster display of detected cars.   Figure 7 shows the grid of 80 pixel wide windows overlapping at a ratio of 0.25 across an example image frame.

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure7.png?raw=true"  width=1000>

<i><u>Figure 7: Visualization of overlapping 80 pixel Sliding Window Search Area</u></i>

---

The sliding windows algorithm was implemented in the function `process_windows()` in [MultiWindow.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/MultiWindow.py).  I attempted to optimize the algorithm with a single run on the HOG gradients for the entire image strip that the sliding windows passed over for each window size.  Implementing this caused me to resize the HOG window size (as described above) to 16x16 so that it was easier to set up specific overlap ratios with this method.

Figures 8, 9 and 10 illustrate the process of car detection;  In figure 8 the SVM has identified overlapping patches as containing car(s).  In Figure 9 the pixels in the overlapping patches have been summed to create a heatmap. In figure 10 a bounding box has been drawn to encompass all pixels with heat values in excess of a threshold value, and this corresponds approximately to the car location(s). 
 
---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure8.png?raw=true"  width=1000>

<i><u>Figure 8: Visualization of Overlapping windows where SVM detected a Car</u></i>

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure9.png?raw=true"  width=1000>

<i><u>Figure 9: Visualization of Heat created by adding overlapping SVM Detections of Cars</u></i>

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10.png?raw=true"  width=1000>

<i><u>Figure 10: Identification of Car Location based on Thresholded Heat Values</u></i>

---

The following images demonstrate where cars were identified on all of the images in the sample folder.  A threshold heat value of 4 was used to discriminate between 

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_1.png?raw=true"  width=1000>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_2.png?raw=true"  width=1000>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_3.png?raw=true"  width=1000>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_4.png?raw=true"  width=1000>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_5.png?raw=true"  width=1000>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_6.png?raw=true"  width=1000>

<i><u>Figure 10: Identification of Car Location based on Thresholded Heat Values</u></i>

---



### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


Note:  For this project I explored using the Atom editor in conjunction with the Hydrogen package. This enabled interaction with the python code within the editor, while I was working to produce self contained python code as opposed to iPython notebooks.

