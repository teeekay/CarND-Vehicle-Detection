###

When using color features and histogram of color, manage 95% accuracy on vehicle classification
Adding in HOG enables 99%+

tried to be as skimpy as possible to reduce computation time.
 - using 6*6 images for color and histogram of colors with HSV color space 16 hist bins.
 - using 12*12 images for HOG with H channel and 3*3 cells get 99.2%
 - using 16*16 images for HOG with H channel and 3*3 cells get 100 %

problems with false positives for all tests on test2.jpg.  switch to YCrCb colorspace and get much better results still have problems on test5.jpg
could try reducing HOGsize and using all channels, or going to specific HOG channel (not Y)
end of June 10

===

C=0.1
4023 cars and 4047 notcars in input training set
Adjusted to 4023 cars and 4023 notcars in training set
4769 cars and 5068 notcars in test set
Adjusted to 4769 cars and 4769 notcars in test set
Run on test set of 1609 samples
shuffled 1609 test samples
2.14 Seconds to train SVC...
Test Accuracy of SVC = 92.7905531385954%
My SVC predicts:
[ 1.  1.  1.  0.  1.  0.  0.  1.  1.  1.  0.  1.  1.  0.  0.  1.  1.  1.
  0.  0.  1.  1.  1.  0.  0.]
For these 25 labels:
[ 1.  1.  1.  0.  1.  0.  0.  1.  1.  1.  0.  1.  1.  0.  0.  1.  1.  1.
  0.  0.  1.  1.  1.  0.  0.]
0.001 Seconds to predict 25 labels with SVC

===
C = 1
4023 cars and 4047 notcars in input training set
Adjusted to 4023 cars and 4023 notcars in training set
4769 cars and 5068 notcars in test set
Adjusted to 4769 cars and 4769 notcars in test set
Run on test set of 1609 samples
shuffled 1609 test samples
2.04 Seconds to train SVC...
Test Accuracy of SVC = 93.53635798632692%
My SVC predicts:
[ 1.  0.  0.  1.  0.  0.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  1.  0.
  0.  0.  0.  0.  0.  1.  1.]
For these 25 labels:
[ 1.  0.  0.  1.  0.  0.  1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  1.  0.
  0.  0.  0.  0.  0.  1.  1.]
0.001 Seconds to predict 25 labels with SVC

===
C=10
4023 cars and 4047 notcars in input training set
Adjusted to 4023 cars and 4023 notcars in training set
4769 cars and 5068 notcars in test set
Adjusted to 4769 cars and 4769 notcars in test set
Run on test set of 1609 samples
shuffled 1609 test samples
2.01 Seconds to train SVC...
Test Accuracy of SVC = 93.7228091982598%
My SVC predicts:
[ 0.  0.  1.  1.  0.  1.  0.  0.  1.  1.  1.  0.  1.  0.  1.  0.  0.  1.
  1.  1.  1.  1.  1.  0.  1.]
For these 25 labels:
[ 0.  0.  1.  1.  0.  1.  0.  0.  1.  1.  1.  0.  1.  0.  1.  0.  0.  1.
  1.  0.  1.  1.  1.  0.  1.]
0.001999 Seconds to predict 25 labels with SVC

===
C=100

4023 cars and 4047 notcars in input training set
Adjusted to 4023 cars and 4023 notcars in training set
4769 cars and 5068 notcars in test set
Adjusted to 4769 cars and 4769 notcars in test set
Run on test set of 1609 samples
shuffled 1609 test samples
2.14 Seconds to train SVC...
Test Accuracy of SVC = 93.22560596643879%
My SVC predicts:
[ 1.  1.  1.  0.  0.  1.  0.  0.  1.  0.  0.  0.  1.  1.  0.  0.  1.  1.
  0.  0.  1.  1.  1.  0.  1.]
For these 25 labels:
[ 1.  1.  1.  0.  0.  1.  0.  1.  1.  0.  0.  0.  1.  1.  0.  1.  1.  1.
  0.  0.  1.  1.  1.  0.  1.]
0.001 Seconds to predict 25 labels with SVC

===
C=1000
4023 cars and 4047 notcars in input training set
Adjusted to 4023 cars and 4023 notcars in training set
4769 cars and 5068 notcars in test set
Adjusted to 4769 cars and 4769 notcars in test set
Run on test set of 1609 samples
shuffled 1609 test samples
2.08 Seconds to train SVC...
Test Accuracy of SVC = 93.16345556246115%
My SVC predicts:
[ 1.  1.  1.  1.  0.  1.  0.  1.  1.  0.  0.  1.  1.  1.  1.  0.  1.  0.
  0.  1.  1.  1.  0.  0.  1.]
For these 25 labels:
[ 1.  1.  1.  1.  0.  1.  0.  1.  1.  0.  0.  1.  1.  1.  0.  0.  1.  0.
  0.  1.  1.  1.  0.  0.  1.]
0.001999 Seconds to predict 25 labels with SVC

==========
{'kernel': 'linear', 'C': 0.1}

Grid scores on development set:

0.973 (+/-0.007) for {'kernel': 'linear', 'C': 0.1}
0.973 (+/-0.007) for {'kernel': 'linear', 'C': 1}
0.973 (+/-0.007) for {'kernel': 'linear', 'C': 10}
0.973 (+/-0.007) for {'kernel': 'linear', 'C': 100}
0.973 (+/-0.007) for {'kernel': 'linear', 'C': 1000}

Detailed classification report:

The model is trained on the full development set.
The scores are computed on the full evaluation set.

             precision    recall  f1-score   support

        0.0       0.99      0.89      0.94       803
        1.0       0.90      0.99      0.95       806

avg / total       0.95      0.94      0.94      1609
