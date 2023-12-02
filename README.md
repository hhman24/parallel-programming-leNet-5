# parallel-programming-leNet-5
Parallel Programming Final Project #HCMUS #CNN #LeNet-5

# Member
|Họ và tên        |MSSV    |    
|:----------------|:------:|
|Hoàng Hữu Minh An|20127102|
|Trần Tiến Hoàng  |20127424|
|Nguyễn Tấn Phát|20127588|

#  Introduction

In this final project, you will be implementing and optimizing the forward-pass of a convolutional layer using CUDA. Convolutional layers are the primary building blocks of convolutional neural networks (CNNs), which are used in many machine learning tasks like image classification, object detection, natural language processing, and recommendation systems. In general, CNNs work well on tasks where the data/input features have some level of spatial relationship.

Your optimized CUDA implementation of the convolutional layer will be used to perform inference for layers C1 and C3. 

We can use mini-dnn-cpp (Mini-DNN) framework as a starting point to implement the modified version of LeNet-5.

We will use the Fashion MNIST dataset, where the inputs to the network will be a single channel images with dimensions of 28 x 28 pixels. The output layer consists of 10 nodes, where each node represents the likelihood of the input belonging to one of the 10 classes (T-shirt, dress, sneaker, boot etc.)

The overall learning objectives for this project are:
 - Demonstrating command of CUDA and optimization approaches by designing and implementing an optimized neural-network convolutional layer forward pass

#  Background knowledge

1 . Convolution Neural Networks (CNNs)
    * Standford [cheatsheat](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#overview)
    * Video CNNs: ["How Convolutional Neural Networks work"](https://www.youtube.com/watch?v=FmpDIaiMIeA)
2 . Lenet-5
    * A Complete Guide: [here](https://www.kaggle.com/code/blurredmachine/lenet-architecture-a-complete-guide/notebook)

# Dataset

[MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), network will be a single channel images with dimensions of 28 x 28 pixels
