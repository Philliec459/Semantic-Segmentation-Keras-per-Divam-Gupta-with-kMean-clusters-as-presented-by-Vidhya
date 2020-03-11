#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:08:34 2020

@author: craig
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps 
import os.path

from keras.models import load_model
from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12




model = pspnet_50_ADE_20K() # in between detail - load the pretrained model trained on ADE20k dataset

#model = pspnet_101_cityscapes() # too much detail - load the pretrained model trained on Cityscapes dataset

#model = pspnet_101_voc12() # Just the People - load the pretrained model trained on Pascal VOC 2012 dataset

#### model = load_model('vgg_unet_1.h5')

# Use any of the 3 pretrained models above



out = model.predict_segmentation(
    inp="sample_images/1_input.jpg",
    out_fname="bed_out.png"
)



# path  
path1 = r'sample_images/1_input.jpg'
# Reading an image in default mode 
image1 = cv2.imread(path1) 
# Window name in which image is displayed 
window_name1 = 'Original image - hit any key exit'   
cv2.imshow(window_name1, image1)  

# path  
path2 = r'bed_out.png'
# Reading an image in default mode 
image2 = cv2.imread(path2) 
# Window name in which image is displayed 
window_name2 = 'Segmented image - hit any key exit'   
cv2.imshow(window_name2, image2)  

def val_shower(im):
    return lambda x,y: '%dx%d = %d' % (x,y,im[int(y+.5),int(x+.5)])

plt.imshow(image2)
plt.gca().format_coord = val_shower(image2)








'''
# =============================================================================
#   ****   Put mouse on image and hit a key to exit ***
# =============================================================================
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# =============================================================================
#   ****   Put mouse on image and hit a key to exit ***
# =============================================================================
'''

label = np.zeros(out.shape )


'''
bedroom image
'''      

label[out ==0] = 0  #walls
label[out ==8] = 8 #window with no drapes
label[out ==28] = 28 #rug
label[out ==15] = 15 #table
label[out ==57] = 57 #edge of pillows
label[out ==39] = 39 #pillows
label[out ==7]  = 77 #bed
label[out ==5]  = 55 #ceiling
label[out ==3]  = 3 #floor and dark parts
label[out ==85] = 85 #fchandalier
label[out ==18] = 99 #windows with drapes
label[out ==3] = 3 #floor
label[out ==36] = 36 #lamps
label[out ==10] = 10 #back wall dresser
label[out ==17] = 79 #plant in corner
label[out ==22 ] = 99 #plant in corne

#label2 = np.array(label)
#y1 = np.maximum(label2)
#y2 = np.minimum(label2)
#x1 = np.maximum(label2)
#x2 = np.minimum(label2)

plt.figure(1)
plt.imshow(out)  #Original Image

plt.figure(2)    
histogram, bin_edges = np.histogram(out, bins=256, range=(0.0, 100))   
plt.title(" Histogram Original Image")
plt.xlabel(" value")
plt.ylabel("pixels")  
plt.plot(bin_edges[0:-1], histogram)  # <- or here



plt.figure(3)
plt.imshow(label)  #Original Image
 
plt.figure(4) 
histogram, bin_edges = np.histogram(label, bins=256, range=(0.0, 100))   
plt.title(" Histogram Labels")
plt.xlabel(" value")
plt.ylabel("pixels")  
plt.plot(bin_edges[0:-1], histogram)  # <- or here
 


