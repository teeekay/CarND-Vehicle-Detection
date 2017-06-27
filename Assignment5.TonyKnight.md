## **Vehicle Detection Project**

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/videograb00002.png?raw=true"  width=1000>

<i><u>Figure 1: Snapshot from Vehicle Detection Video</u></i>

---


### Writeup by Tony Knight - 2017/06/27 

## Training

### Training Set

I decided to use the GTI data set as the basis for training the Support Vector Machine Classifier.  However, this data set was slightly imbalanced, with more images which were "non-vehicles" than images of vehicles.  I supplemented the car dataset with 1200 images of cars from the KITTI dataset.   I also added a few samples of "curated" non-car images taken from the project video in order to attempt to train the classifier away from some false positive hits.  This resulted in a total of 4023 images of cars and 4047 images of non-cars (which I trimmed to 4023) in the training dataset .    

For the test dataset I used images of cars from the KITTI dataset (excluding those used in the training set) and non-car images from the extras dataset (taken from the video).  I took a random sample of approximately 800 images (20% of the size of the training set) from each category in the test set to evaluate the performance of the classifier.  Figure 2 shows images of "cars" and "non-cars" used in the training and test sets.

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure2.png?raw=true"  width=700>

<i><u>Figure 2: Images of Cars and Non-cars used in training and Test sets</u></i>

---

### Training Features

I decided to use a combination of the color and histogram of gradients (HOG) features to characterize each image patch.  I wanted to make the size of the patches as small as possible in  order to speed up processing during image analysis.  I decided that I would resize each image patch to 16x16 pixels to extract the HOG, and 8x8 pixels to extract color information.  Based on initial tests with the test images I decided to use the YCrCb color space, and to use all 3 of the color channels in the HOG analysis.

#### Histogram of Oriented Gradients (HOG)

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure5.png?raw=true"  width=700>

<i><u>Figure 3: Image of Car and HOG visualization produced from Y color channel of image in YCrCB colorspace</u></i>

---

The code to run the HOG analysis is contained in function `extract_features()` and in function `get_hog_features()` in [feature_extraction.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/feature_extraction.py) on lines 94 through 117, and lines 5 to 16 respectively.  The scikit implementation of `hog()` was called from `get_hog_features()` to calculate the HOG features.

I tried various combinations of parameters and ended up using 4 pixels per side on cells, and using 2x2 cells in a block.  This resulted in a total of 3x3 blocks of 2X2 cells of histograms of gradients from each image.  Part of the reason for selecting this size, was it facilitated implementation of a sliding window with overlap ratios of 0.25, or 0.5 when only calculating HOGs once on the whole image (as 4 blocks are passed over (3 in the patch and one on the boundary of the patch) when sliding a window across the width of a patch in one direction).

I chose to use 9 bins for the orientations of the histograms as it produced good results and using more bins did not improve results.

The selection of these parameters resulted in 3x3x2x2x9=324 HOG features for each  of the 3 color channels on the image patch (972 total). 

I set transform_sqrt to True, (normalizing features before calculating HOG) even though this slightly increased the HOG calculation time, as it appeared to produce better results when I tested the implementation on the video.

#### Color features

Using a patch size of 8x8 pixels, the color distribution for each of the Y, Cr and Cb color channels was split between 16 histogram bins in function `color_hist()` in [feature_extraction.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/feature_extraction.py) (lines 32 to 45)  and produced 48 features.  The value of each pixel in each color channel was also unraveled in function `bin_spatial()` (lines 19 to 29) and produced 3x8x8 = 192 features.

#### Scaler

The values of the 3 types of features (Color Histogram, Color Distribution, HOG) were normalized using Scikit's `StandardScaler()` which was fit to the training set on line 99 of [combofeatures.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/combofeatures.py).  Figure 4 shows images of cars and non-cars in the training set, and the resulting plots of features before and after normalization. 

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure3.png?raw=true"  width=700>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure4.png?raw=true"  width=700>


<i><u>Figure 4: Images of Car and non Car patches and resulting raw and scaled feature sets</u></i>

---


### Training Classifier

Scikit's `LinearSVC()` SVM classifier was trained and tested in [combofeatures.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/combofeatures.py) between lines 165 and 176.  I adjusted the value of the C hyperparameter between 0.1 and 1000 in steps of orders of magnitude, but found there was little change in accuracy results on the test set.  The accuracy rates ranged between a low of 92.7 when C was 0.1 and a high of 93.7 when C was set at 10, and back down to 93.1 at C=1000 (The results are shown in Figure 5)

The error rate generally was due to false positives for detection of cars on images that weren't cars.  Almost all car images were identified as cars.

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure6.png?raw=true"  width=400>

<i><u>Figure 5: Plot of Accuracy results as log10 C is varied</u></i>

---


### Sliding Window Search

I initially set up a sliding image search with 5 sizes of windows (edge lengths 40 pixels, 80 pixels, 120 pixels and 200 pixels).  I ran tests with overlap ratios of 0.5 and 0.25, and found that 0.25 produced better results.  After experimentation, and specifically after visualizing the response when using each size window, I realized that I was able to obtain the best results when using 80 pixel windows alone.  

Use of the smaller 40 pixel wide windows did not detect smaller cars on the horizon, but required a large amount of processing time, so I stopped using them.

I found that the additional use of windows larger than 80 pixels width added more false positives, but did not really add any better detection of cars that were not picked up by the 80 pixel windows.  The program attempts to limit displaying false positives in the videostream by using a buffer of heat image frames, and thresholding for pixels that have had heat for a specified number of frames.  However, when using the larger windows, I found that large areas remained "hot" for long periods in areas where false positives were detected. The history buffer threshold had to be increased to a relatively large threshold (up to 15 frames) to eliminate these false positives.  This had negative effect of causing a significant delay in displaying valid detection of new cars.

Removal of the larger window sizes enabled the history threshold to be reduced down to 5 frames, which resulted in much faster response to detected cars.

Figure 6 shows the grid of 80 pixel wide windows overlapping at a ratio of 0.25 across an example image frame.

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure7.png?raw=true"  width=700>

<i><u>Figure 6: Visualization of overlapping 80 pixel Sliding Window Search Area</u></i>

---

The sliding windows algorithm was implemented in the function `process_windows()` in [MultiWindow.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/MultiWindow.py).  I attempted to optimize the algorithm with a single run on the HOG gradients for the entire image strip that the sliding windows passed over for each window size.  Implementing this feature caused me to resize the HOG window size (as described in the HOG features section above) to 16x16 so that it was easier to set up specific overlap ratios with this method.

Figures 7, 8, 9, and 10 illustrate the process of car detection with sliding windows;  In Figure 7, the SVM has identified overlapping patches as containing car(s).  In Figure 9, the pixels in the overlapping patches have been summed to create a heatmap. In figure 10, a bounding box has been drawn to encompass all contiguous pixels with heat values in excess of a threshold value, corresponding approximately to the car location(s). 
 
---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure8.png?raw=true"  width=700>

<i><u>Figure 7: Visualization of Overlapping windows where SVM detected a Car</u></i>

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure9.png?raw=true"  width=700>

<i><u>Figure 8: Visualization of Heat created by adding overlapping SVM Detections of Cars</u></i>

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure9a.png?raw=true"  width=700>

<i><u>Figure 9: Visualization of Binary Heat Map after applying Thresholds</u></i>

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_1.png?raw=true"  width=700>

<i><u>Figure 10: Identification of Car Location based on Thresholded Heat Values</u></i>

---

The following images demonstrate how the process was able to correctly identify the cars on all of the images in the test_images folder without false positives.  A threshold heat value of 4 was used to discriminate against false positives (a history threshold of 1 was used as only one frame was analyzed).

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_1.png?raw=true"  width=500>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_2.png?raw=true"  width=500>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_3.png?raw=true"  width=500>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_4.png?raw=true"  width=500>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_5.png?raw=true"  width=500>
<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure10_6.png?raw=true"  width=500>

<i><u>Figure 11: Identification of Car Locations in images in test_images directory</u></i>

---


### Video Implementation


Here's a link to my [project_video](./project_video_outputA.mp4). The code to generate the video is called from [videoprocessor.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/videoprocessor.py) and in [image_screener.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/image_screener.py).  I was also able to add in my code from the lane detection project to produce a combined output of lane detection and vehicle detection.

#### Filtering strategy for video

The combination of the hard edge on the median wall, which occasionally generates false positives on its own, and the partial views of cars above the median wall combine to create heat values above the threshold value.  

I implemented a circular buffer (lines 111 to 137 of [image_screener.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/image_screener.py)) into which I put frames of the heat values generated so that I could threshold the heat values based on the numbers of "hits" on each pixel in the buffer.  For example, I could set up the buffer to hold 20 frames, and threshold pixels that had hits on 15 of the frames in the buffer.  I found this level of "history threshold" was required when I was using multiple size windows.  This resulted in a more than 0.5 second delay after a car was first seen before it could be displayed, which I did not consider to be optimal.

I adjusted the buffer so that I could store heat values obtained individually for each size window used.  This allowed me to display the heat generated from each window size at the right of the output video to better visualize what was going on.  Once I did this, I realized that I only really needed to use the one size of window (80 pixels per side).  When I switched to using only size 80 pixel windows, I found that the history threshold could be reduced to 5 frames in a 7 frame buffer to get relatively good results with only occasional false positives at the left side of the screen.  

An issue in this implementation of the history threshold is that when the car goes over a bump (22.5s to 24 s in the video), all the objects are displaced rapidly in the image for a few frames, which causes the history threshold to prevent detected objects from being displayed.  The history threshold may also prevent oncoming cars from being displayed well as they could move across the screen too fast to allow detection in the same area of the screen on subsequent frames.

When determining the appropriate heat threshold to use, I had to consider both eliminating false positives, but also maintaining detection of faint positives (e.g. the white car at t=24 to 30s) for as long as possible.  The optimal heat threshold was between 4 and 5, and I selected 5 which worked relatively well, but did eliminate the white car detection in some sections of the video where it was detected at a value of 4. 

I have thought of other possible strategies to improve overall detection (including tying the heat threshold to the vertical position so that the threshold is decreased as the item moves up the screen (and becomes smaller)).

#### Bounding Boxes

The functions to create and threshold heatmaps and then generate bounding boxes around the label instances are located in [image_screener.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/image_screener.py) between lines 32 and 92. As discussed above, the heat values are calculated by incrementing the value at each pixel for every window it is contained in that the SVM identifies as being a car.  In each frame, these values are thresholded so values below the threshold are set to 0.  All the remaining values are set to 1.  The sum of pixels across the multiple frames in the buffer is calculated and this is used to threshold the heat values so any pixels where the buffer total is less than the history_threshold are also set to zero.  This produces a binary image as shown in figure 12.  

---

<img src="https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/output_images/figure9a.png?raw=true"  width=700>

<i><u>Figure 12: Binary Thresholded Image of Heat Values</u></i>

---

SciPy's `ndimage.measurements.label()` function is called to locate the separate contiguous instances of thresholded values in the binary image.  The `draw_labeled_bboxes()` function on lines 68 to 92 of [image_screener.py](https://github.com/teeekay/CarND-Vehicle-Detection/blob/master/image_screener.py) extracts the top left and bottom right co-ordinates of the labelled areas, and then draws bounding boxes.

For this video, the function was optimized to draw the boxes in different colors, and to attempt to maintain the same object box in the same color for as long as possible.  The `label()` function produces a list sorted from the top to bottom of the image.  Since the main stable targets in the video were located on the right side of the image, the box locations were re-sorted by their right hand co-ordinates to make boxes at the right side of the image the first targets drawn.


#### Classifier Improvement

The classifier often detected false positives on patches containing the yellow lines and median barrier (possibly in combination with one or two distinct spots) at the left side of the lane.  I attempted to improve performance by cropping a set of images from these areas (in space and time) of the video and adding them to the training set as non-car images.  I did not see as much response from the yellow lines after this, but the median barrier continued to trigger the classifier.  I may not have added enough images of the top of the median at the correct scale to bias the classifier against this image.

As discussed above, in the training section, I explored the effect of adjusting the Hyperparameter value used in the classifier, and adjusted the value of C to 10 from the default of 1 for a small improvement in accuracy.

### Discussion

#### Speed of Implementation

I found that on my computer, the car detection algorithm took approximately 0.12 seconds per frame which is about 1/6th the speed required (although when combined with the lane detection it took about 0.5 secs per frame).  I think the code could be optimized and rewritten in C to run fast enough to produce real-time output.  However, convolutional neural network approaches like [YOLO](https://pjreddie.com/darknet/yolo/) and [YOLT](https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-38dad1cf7571) look like they would be more flexible and just as fast (especially if running on GPU like hardware).

#### Training Set

Although there was a relatively large image set to use when training the SVM, it appeared that the set did not cover all image conditions, and may have included more dark than light cars.  I think that reduced false positives would have been produced with a larger variety of non-car images.

#### Threshold Values

As discussed above, the use of a heat threshold value and (possibly the history threshold value) which varies according to the position on the image frame could be explored to evaluate if it would enable better detection of cars further up the image, and better response to oncoming cars at the left of the image.  

#### Object Bounding Box

Use of the history threshold enabled a small amount of stabilization of the detected car shape.  It was still quite jittery.  This could be improved by storing the shape for each frame, and averaging it over multiple frames.  A better solution would probably involve maintaining dimensions of objects combined with tracking the associated center of mass of the object in each frame.  This could be used to better track the motion of the object, and differentiate between different objects.  This could be further enhanced to implement models of the detected objects with the following characteristics (size, velocity, direction, distance).

Models of expected car motions could attempt to accommodate the following four situations that I think should be expected and would enable the system to assess risk from the objects:

a) cars parallel to the car at relatively slow speeds compared to the car,

b) stationary cars (e.g. parked at the side of the road),

c) oncoming cars moving towards the car at high speed,

d) cars moving perpendicularly to the car (e.g. from a side street).   


### Additional Notes

For this project I explored using the Atom editor in conjunction with the Hydrogen package (which connects to a Jupyter kernel) for the first time.  This enabled me to interact fairly effectively with the python code within the editor, while I was working to produce self contained python code (as opposed to iPython notebooks).
