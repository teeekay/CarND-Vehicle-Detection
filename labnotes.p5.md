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