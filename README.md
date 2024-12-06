# Computer Vision Notebooks

This repository contains a collection of Jupyter notebooks focused on various aspects of computer vision. Each notebook covers different techniques and methodologies used in the field, providing both theoretical insights and practical implementations.

## Table of Contents

- [Intro to Computer Vision](#intro-to-computer-vision)
- [Operation Types](#operation-types)
- [Edge Detection and Morphological Operations](#edge-detection-and-morphological-operations)
- [Contours and Detection](#contours-and-detection)
- [Harris and SIFT](#harris-and-sift)
- [Bag of Visual Words (BOW)](#bag-of-visual-words-bow)
- [YOLO](#yolo)
- [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
- [Face Detection](#face-detection)
- [Semantic Segmentation](#semantic-segmentation)
- [Video Tracking](#video-tracking)

## Intro to Computer Vision

This notebook covers foundational concepts in computer vision, including:

1. **Review Linear Algebra and Vector Calculus**: Essential mathematical concepts for understanding image processing.
2. **Basic Image Processing**: Techniques for reading/writing images from/to files and displaying them.
3. **Skin Detection**: Methods for detecting skin regions in images.
4. **Red Eye Detection**: Techniques for identifying and correcting red-eye effects in photographs.

## Operation Types

This notebook explores different types of image operations:

1. **Point Operations**: Operations that affect individual pixels.
2. **Neighborhood Operations**: Operations that consider the surrounding pixels.
3. **Geometric Operations**: Transformations that change the geometry of images.

## Edge Detection and Morphological Operations

This notebook focuses on advanced image processing techniques:

- **Logical Operations**: Basic operations like AND, OR, NOT applied to images.
- **Histograms**: Analyzing the distribution of pixel intensities.
- **Thresholding**: Techniques for segmenting images based on intensity values.
- **Morphological Operations**: Operations that process images based on their shapes.
- **Gradients**: Techniques for detecting changes in intensity.
- **Edge Detection**: Methods for identifying edges within images.

## Contours and Detection

This notebook covers techniques for detecting shapes and objects in images:

- **Contours**: Finding and analyzing the outlines of shapes.
- **Connected Components Labeling**: Identifying connected regions in binary images.
- **Object Detection**: Techniques for detecting specific objects within images.
- **Template Matching**: Finding parts of an image that match a template.
- **HOG (Histogram of Oriented Gradients)**: A feature descriptor used for object detection.
- **SVM (Support Vector Machine)**: A classification technique used in conjunction with HOG.
- **Image Pyramids**: Techniques for multi-scale image processing.

## Harris and SIFT

This notebook focuses on feature detection and description:

- **Descriptors**: Techniques for describing image features.
- **Detectors**: Methods for detecting key points in images, including Harris corner detection.
- **Panoramas with SIFT**: Using SIFT (Scale-Invariant Feature Transform) for stitching images together.

## Bag of Visual Words (BOW)

This notebook introduces the concept of Bag of Visual Words:

- **Bag of Visual Words**: A method for representing images based on visual features.
- **K-means Clustering**: A clustering technique used to create visual words.

## YOLO

This notebook covers object detection using the YOLO (You Only Look Once) framework:

- **Object Detection with YOLO**: Techniques for real-time object detection in images and videos.

## Generative Adversarial Networks (GANs)

This notebook explores GANs:

- **Working with MNIST**: Using GANs to generate handwritten digits from the MNIST dataset.

## Face Detection

This notebook focuses on techniques for detecting faces in images:

- **Viola-Jones**: A popular algorithm for face detection.
- **Deep Learning Approaches**: Modern techniques for face detection using deep learning.


## Semantic Segmentation

This notebook covers semantic segmentation using two models:

- **FCN**: Fully Convolutional Network with ResNet101 as a backbone
- **Deelabv3**:  SOTA model from google 

## Video Tracking

This notebook covers object trackign in video:
- **Tracking single object**: using Opencv based functions
2. **Tracking multiple objects**: via SSD


## Conclusion

These notebooks provide a comprehensive introduction to various computer vision techniques and methodologies. They are designed for both beginners and experienced practitioners looking to enhance their understanding of the field.
