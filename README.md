# Basic-image-segmentation-keras-with-kMean-clusters-as-pesented-by-Vidhya
kMean clusters used for final labeling of segments and the repository of Divamgupta.

https://github.com/divamgupta/image-segmentation-keras

This repository is based on studying Divamgupta's GitHub repository, blog and other written materials  that I have found online. This is great work and a real help to a novice such as myself. I will be creating a few repositories using my interpretation of how this all might be implemented. I might be wrong, so any constructive criticism is welcome. We are still using Divamgupta's pre-trained example shown in his Readme file. This is the original photo that we are working from:

![Image](1_input.jpg)

We use the pretrained code to create segmentation items from a photo using load_pretrain_and_create_image.py driven from an xterm. We then used interactive_plots_clusters_with_box.py to isolate the various predicted clusters;

![Image](cluster_pic.png)

and then interactively select the pixel value associated with a particular cluster or segment in the photograph that we are trying to isolate. The python program is interactive_plots_clusters_with_box.py 

We made our initial segmentation working with the the bedroom picture under the sample_images subdirectory (1_input.jpg). After discriminating all of the major features in the photo as shown above, we then isolate a particular item in the picture (bed), and create a subsequent image showing just that feature with a green rectangle around it. 


![Image](bed_cluster_labels_box.png)

We are working in Ubuntu and each python .py program is driven from an xterm using 'python xxxx.py' as the command. 

This second repository is similar to another Keras application repository except we are be using the kMean clusters for image segmentation as presented by Vidhya for image segmentation. 


