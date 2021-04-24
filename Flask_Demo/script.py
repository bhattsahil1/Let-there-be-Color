#!/usr/bin/env python
import cv2
import os
import argparse
from skimage import io, color

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

img_final = cv2.resize(image, (224, 224))

#path = r'/home/jayati/project-6-6/Flask_Demo/static/'
#os.chdir(path)
cv2.imwrite('test_image_1.jpg', img_final)
#path_2 = r'/home/jayati/project-6-6/Flask_Demo/'
#os.chdir(path_2)

