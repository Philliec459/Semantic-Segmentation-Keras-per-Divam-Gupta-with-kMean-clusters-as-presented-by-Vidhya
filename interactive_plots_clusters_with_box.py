#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:37:19 2020

@author: craig
"""


'''
# =============================================================================
# Drive this from terminal window
# =============================================================================
'''
from PIL import Image
import os.path

from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy import ndimage
from skimage.io import imsave


from skimage.filters import sobel,gaussian,hessian,frangi,laplace,median
from skimage.color import rgb2gray

from sklearn.cluster import KMeans



pic = cv2.imread('bed_out.png')/255
#pic = plt.imread('bed_out.jpg')/255  # dividing by 255 to bring the pixel values between 0 and 1
print()
print('Original Shape of Picture', pic.shape)

#plt.figure(0)
#plt.imshow(pic)

pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
print()
print('New Shape for Picture (pic_n) after being flattened', pic_n.shape)

'''
              Predict Clusters
      fit(pic_n) which is the flattened pic image
'''
kmeans = KMeans(n_clusters=20, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

'''
     reshape new  pic2show and call it cluster_pic 
'''
cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
print()
print('New Shape for Picture (cluster_pic) after being UN-flattened', cluster_pic.shape)

print()
print('Full set of kMeans cluster centers arrays:')
print(kmeans.cluster_centers_)

# save new clusters for chart
y_km = kmeans.fit_predict(pic2show)


print()
print('original shape of y_km fit_predicted:', y_km.shape)
#print('pic.shape 0, 1 and 2: (', pic.shape[0], pic.shape[1], pic.shape[2], ')')
y_km_image = y_km.reshape(pic.shape[0], pic.shape[1])
print()
print('New shape of y_km_as an image', y_km_image.shape)
print()
print()

data =np.around(median(rgb2gray(cluster_pic*1))) #averagint


'''
     **** Create interactive Plot ****
'''
print()
print('Create interactive Plot')
print()

#import mpldatacursor
from mpldatacursor import datacursor

fig, ax = plt.subplots()
#ax.imshow(cluster_pic, interpolation='none')
#ax.imshow(data, interpolation='none')
ax.imshow(y_km_image, interpolation='none')

#mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'))
datacursor(display='single')
plt.show()




'''
 Create labels for labeled image that will be saved  
'''
print()
print('Create Labels for plot to be saved')
print()


label = np.zeros(y_km_image.shape )
#label[out ==0] = 0  #walls
#label[out ==8] = 8 #window with no drapes

#label[y_km_image ==4] = 0.6 #rug
#label[y_km_image >14 & y_km_image <15] = .95 #most of the boundaries
#label[y_km_image <16] = .95 #most of the boundaries

label[y_km_image ==4]  = .9 #bed
#label[y_km_image ==4] = 0.6 #rug

#label[out ==15] = 15 #table
#label[out ==57] = 57 #edge of pillows
#label[out ==39] = 39 #pillows
#label[y_km_image ==2]  = .8 #bed
#label[out ==5]  = 55 #ceiling
#label[out ==3]  = 3 #floor and dark parts
#label[out ==85] = 85 #fchandalier
#label[out ==18] = 99 #windows with drapes
#label[out ==3] = 3 #floor
#label[out ==36] = 36 #lamps
#label[out ==10] = 10 #back wall dresser
#label[out ==17] = 79 #plant in corner
#label[out ==22 ] = 99 #plant in corne




'''
 Save labeled image
'''
print()
print('Save Labeled Image')
print()

#### for images scaled 0 to 1
img_out1=os.path.join('segs/' + 'cluster_pic.png')
#############im = Image.fromarray(label)
############im = label.convert("L")
#imsave(img_out,cluster_pic)
imsave(img_out1,cluster_pic)


#### for images scaled 0 to 1
#img_out=os.path.join('segs/' + 'bed_labels.png')
img_out2=os.path.join('segs/' + 'bed_cluster_labels.png')
#############im = Image.fromarray(label)
############im = label.convert("L")
#imsave(img_out,cluster_pic)
imsave(img_out2,label)

   
'''

         This is the green Box

'''


pic = cv2.imread(img_out2)
img = cv2.pyrDown(pic)


# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                127, 255, cv2.THRESH_BINARY)
# find contours and get the external one
contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
#                cv2.CHAIN_APPROX_SIMPLE)
# with each contour, draw boundingRect in green
# a minAreaRect in red and
# a minEnclosingCircle in blue
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)

    '''
    # draw a green rectangle to visualize the bounding rect
    '''
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # get the min area rect
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)



#    '''
#    # draw a red 'nghien' rectangle
#    '''
#    cv2.drawContours(img, [box], 0, (0, 0, 255))



#    '''
#    # finally, get the min enclosing circle
#    '''
#    (x, y), radius = cv2.minEnclosingCircle(c)
#    # convert all values to int
#    center = (int(x), int(y))
#    radius = int(radius)
#    # and draw the circle in blue
#    img = cv2.circle(img, center, radius, (255, 0, 0), 2)



print(len(contours))
cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
cv2.imshow("Hit Esc to leave", img)
cv2.imshow("Hit Esc to leave", img)


'''
     **** use Esc to get out ****
'''
while True:
    key = cv2.waitKey(1)
    if key == 27: #ESC key to break
        break

cv2.destroyAllWindows()

img_out3=os.path.join('segs/' + 'bed_cluster_labels_box.png')
img = cv2.pyrUp(cv2.drawContours(img, contours, -1, (255, 255, 0), 1))
imsave(img_out3,img)
#imsave(img_out3,cv2.drawContours(img, contours, -1, (255, 255, 0), 1))