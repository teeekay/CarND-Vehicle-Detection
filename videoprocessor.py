import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from image_screener import find_cars, car_finder, draw_labeled_bboxes, apply_threshold
from lanelines import data_storage, feed_the_beast
from scipy.ndimage.measurements import label
#%matplotlib inline

cap = cv2.VideoCapture('./project_video.mp4')

fps = float(cap.get(cv2.CAP_PROP_FPS))
framewidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video opened with framecount of {:4,d}, dimensions ({:4d},{:4d}), and speed of {:3.03f} fps."
     .format(framecount, framewidth, frameheight, fps))

# skip ahead to area with cars in it
#cap.set(cv2.CAP_PROP_POS_FRAMES, 175)


#fourcc = cv2.VideoWriter_fourcc(*"H264")
fourcc = cv2.VideoWriter_fourcc(*"DIVX")

# space for four frames showing heatmaps
framewidth = framewidth+framewidth//4


video_filename='project_video_5in7over3_80_X.25.mp4'
out = cv2.VideoWriter(video_filename, fourcc, fps, (framewidth, frameheight))
print("Writing to video file {}".format(video_filename))

frames = 0
#landfinding
#initialize place to keep old and new binary thresholded images
data_stored = data_storage()
data_stored.add_fps(fps)

#initialize vehicle finding algo
car_values = car_finder()
car_values.set_overlap(0.25)


t0 = time.time()
t1 = time.time()

total_heat_threshold = 160
history_heat_threshold = 3
history_threshold = 5

while(cap.isOpened()):
    frames += 1
    #lanefinding
    data_stored.set_frame(frames)


    #uncomment for early stop
#    framecount = 200

    if frames > framecount:
        print("\nClosed video after passing expected framecount of {}".format(frames-1))
        break
    ret, image1 = cap.read()
    if ret == True:
        #lanefinding
        #output = feed_the_beast(image1, data_stored, convolve=True)
        output=image1

        draw_img, boxes, histheat, heat = find_cars(image1, car_values, history_heat_threshold, history_threshold)
        #heat = apply_threshold(car_values.getheatval(),total_heat_threshold)
        heat = car_values.getlastheatval()[:,:,0] #only using first layer for test
        heat[heat<3]=0
        #labels = label(heat)
        #output = draw_labeled_bboxes(output,labels)

        msecs = float(cap.get(cv2.CAP_PROP_POS_MSEC))



        #image1BGR = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
        #image1BGR = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        #out.write(image1BGR)
        image = np.zeros((frameheight,framewidth,3),dtype=np.uint8)
        #image[0:720,0:1280,:] = output
        image[0:720,0:1280,:] = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
        image[0:180,1280:,2] = cv2.resize(heat[:,:]*20, (320,180), interpolation=cv2.INTER_AREA)
        #image[180:360,1280:,1] = heat[:,:,1]*20
        #image[360:540,1280:,2] = heat[:,:,2]*20
        #image[540:720,1280:,:] = np.sum(heat,axis=0)*10
        out.write(image)
        t2=time.time()
        print("Frames: {0:02d}, Seconds: {1:03.03f} Processing Time per frame: {2:03.03f} s, Total Processing Time: {3:05.03f}s".format(frames, frames/fps, t2-t1, t2-t0), end='\r')
        t1=time.time()
    else:
        print("\nClosed video after getting empty frame {}".format(frames))
        break

cap.release()
out.release()