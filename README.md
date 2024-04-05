# Alohomora 

### RBE 549: Classical and Deep Learning Approaches for Geometric Computer Vision [[Problem Statement 1](https://rbe549.github.io/spring2024/hw/hw0/)]



## What is this about?

#### Phase 1: Classical Approach

The task of boundary detection in computer vision seeks to identify where one object transitions to another within an image, a challenge complicated by the need for object-specific reasoning. Classical edge detection algorithms like Canny and Sobel focus on finding intensity discontinuities, but the more advanced probability of boundary (pb) algorithm enhances detection by also considering texture and color discontinuities. We explore a simplified version of the pb algorithm, which analyzes brightness, color, and texture information at multiple scales to predict per-pixel boundary probabilities. Our approach, drawing from methods detailed in recent Berkeley research, demonstrates significant improvements over classical methods by reducing false positives in textured regions. This is validated through qualitative comparisons with human annotations from the Berkeley Segmentation Data Set 500 (BSDS500), underscoring our simplified detector's enhanced accuracy in boundary detection.



#### Phase 2: Deep Learning Approach

For Phase 2, we implemented multiple neural network architectures and compared them on various criterion like number of parameters, train and test set  accuracies and provide detailed analysis of why one architecture works  better than another one. CIFAR-10 is a dataset consisting of 60000, 32×32 colour images in 10 classes, with 6000 images per class is used. There are 50000 training images and 10000 test images.



## Acknowledgment 

This project is done as a requirement in the course RBE 549 taught by Dr. [Nitin J. Sanket](https://nitinjsanket.github.io/) at the Worcester Polytechnic Institute.  This is a fun and challenging way to learn some basic concepts! 

Please read the report! Thank you!
