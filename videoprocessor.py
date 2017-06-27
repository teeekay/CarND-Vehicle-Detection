import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from image_screener import find_cars, car_finder, draw_labeled_bboxes, apply_threshold
from lanelines import data_storage, feed_the_beast
from scipy.ndimage.measurements import label

cap = cv2.VideoCapture('./project_video.mp4')
# get details on the video being processed
fps = float(cap.get(cv2.CAP_PROP_FPS))
framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video opened with framecount of {:4,d}, dimensions ({:4d},{:4d}), and speed of {:3.03f} fps."
     .format(framecount, framewidth, frameheight, fps))

# for debugging we can skip ahead to area with cars in it
#cap.set(cv2.CAP_PROP_POS_FRAMES, 175)


#fourcc = cv2.VideoWriter_fourcc(*"H264")
fourcc = cv2.VideoWriter_fourcc(*"DIVX")

# Add space for four frames showing heatmaps at right of video
framewidth = framewidth+framewidth//4

# set up the output file
video_filename='project_video_output.mp4'
out = cv2.VideoWriter(video_filename, fourcc, fps, (framewidth, frameheight))
print("Writing to video file {}".format(video_filename))

frames = 0

# lanefinding
# initialize lane finding object to store details including
# old and new binary thresholded images
data_stored = data_storage()
data_stored.add_fps(fps)

# initialize vehicle finding object
car_values = car_finder()
# explicitly set overlap ratio to .25
car_values.set_overlap(0.25)

#set up timers to record length of time per frame and overall time to process video
t0 = time.time()
t1 = time.time()

# threshold of heat value in current frame required to register a hit
heat_threshold = 3

# threshold of number of frames within recent history that pixel needs to exceed
# heat threshold for heat to be valid
history_threshold = 5

# threshold to use for total sum of heat values over history of n frames (not used)
#total_heat_threshold = 160

while(cap.isOpened()):
    frames += 1
    #lanefinding
    data_stored.set_frame(frames)

    #uncomment for early stop
    #framecount = 200

    if frames > framecount:
        print("\nClosed video after passing expected framecount of {}".format(frames-1))
        break
    ret, image1 = cap.read()
    if ret == True:
        # lanefinding call - returns BGR image with lanefinding and other details on it
        output = feed_the_beast(image1, data_stored, convolve=True)
        #output=image1

        #output, boxes, histheat, heat = find_cars(image1, car_values, history_heat_threshold, history_threshold)
        find_cars(image1, car_values, heat_threshold, history_threshold, external_markup=True)

        # return heat values for each pixel
        # only using first layer since we are only using dimension 80 windows
        heat = car_values.getlastheatval()[:,:,0]
        heat[heat<heat_threshold]=0

        # returns count of frames in buffer that each pixel exceeds threshold
        historycount = car_values.getheat()
        # get rid of heat where there is not enough history
        heat[historycount<history_threshold]=0
        labels = label(heat)
        #now draw the boxes on the image
        output = draw_labeled_bboxes(output,labels)

        # set up wider output image to accomodate heat dispalys on the side
        image = np.zeros((frameheight,framewidth,3),dtype=np.uint8)


        image[0:720,0:1280,:] = output
        image[0:180,1280:,2] = cv2.resize(heat[:,:]*40, (320,180), interpolation=cv2.INTER_AREA)
        cv2.putText(image, "Heatmap - 80 pixel windows", (1300,25), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.rectangle(image, (1280,0), (1600,180), (255,255,255), 1)
        #did not use these because found that using windows with dimension 80 alone worked best
        #image[180:360,1280:,1] = heat[:,:,1]*20
        #image[360:540,1280:,2] = heat[:,:,2]*20
        #image[540:720,1280:,:] = np.sum(heat,axis=0)*10
        out.write(image)

        t2=time.time()
        msecs = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        print("Frames: {0:02d}, Seconds: {1:03.03f} Processing Time per frame: {2:03.03f} s, Total Processing Time: {3:05.03f}s".format(frames, frames/fps, t2-t1, t2-t0), end='\r')
        # start the frame timer again
        t1=time.time()
    else:
        print("\nClosed video after getting empty frame {}".format(frames))
        break

cap.release()
out.release()
