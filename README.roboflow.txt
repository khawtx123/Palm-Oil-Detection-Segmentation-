
ffb oil palm 1 - v17 2021-09-24 3:44am
==============================

This dataset was exported via roboflow.com on April 18, 2023 at 7:32 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 155 images.
Oil-palm are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)
* Auto-contrast via contrast stretching

The following augmentation was applied to create 1 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random brigthness adjustment of between -30 and +30 percent
* Random exposure adjustment of between -20 and +20 percent


