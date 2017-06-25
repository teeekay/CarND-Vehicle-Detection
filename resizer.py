import numpy as np
import cv2
import glob

notcars = glob.glob('./non-vehicles/curated/test/unt*.png')
size = (64,64)

i = 21
for img in notcars:
  i += 1
  name = "./non-vehicles/curated/resized"+str(i)+".png"
  image = cv2.imread(img)
  print("processing {} to size {} in file {}".format(img,size, name))
  image = cv2.resize(image, size)
  retval = cv2.imwrite(name, image)
  print("saving - {}".format(retval))
