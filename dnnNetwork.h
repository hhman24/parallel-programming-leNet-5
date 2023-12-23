/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 *
 * Modified for the ECE 408 class project
 * Fall 2020
 */

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <cstdlib>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/conv_gpu.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

Network dnnNetwork_CPU()
{
    Network dnn1;
    Layer *conv1 = new Conv(1, 28, 28, 6, 5, 5);
    Layer *pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *conv2 = new Conv(6, 12, 12, 16, 5, 5);
    Layer *pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer *fc1 = new FullyConnected(pool2->output_dim(), 120);
    Layer *fc2 = new FullyConnected(120, 84);
    Layer *fc3 = new FullyConnected(84, 10);
    Layer *relu_conv1 = new ReLU;
    Layer *relu_conv2 = new ReLU;
    Layer *relu_fc1 = new ReLU;
    Layer *relu_fc2 = new ReLU;
    Layer *softmax = new Softmax;
    dnn1.add_layer(conv1);
    dnn1.add_layer(relu_conv1);
    dnn1.add_layer(pool1);
    dnn1.add_layer(conv2);
    dnn1.add_layer(relu_conv2);
    dnn1.add_layer(pool2);
    dnn1.add_layer(fc1);
    dnn1.add_layer(relu_fc1);
    dnn1.add_layer(fc2);
    dnn1.add_layer(relu_fc2);
    dnn1.add_layer(fc3);
    dnn1.add_layer(softmax);

    // loss
    Loss *loss = new CrossEntropy;
    dnn1.add_loss(loss);
    return dnn1;
}

Network dnnNetwork_GPU()
{
    Network dnn2;
    Layer *conv1 = new Conv_GPU(1, 28, 28, 6, 5, 5);
    Layer *pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *conv2 = new Conv_GPU(6, 12, 12, 16, 5, 5);
    Layer *pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer *fc1 = new FullyConnected(pool2->output_dim(), 120);
    Layer *fc2 = new FullyConnected(120, 84);
    Layer *fc3 = new FullyConnected(84, 10);
    Layer *relu_conv1 = new ReLU;
    Layer *relu_conv2 = new ReLU;
    Layer *relu_fc1 = new ReLU;
    Layer *relu_fc2 = new ReLU;
    Layer *softmax = new Softmax;
    dnn2.add_layer(conv1);
    dnn2.add_layer(relu_conv1);
    dnn2.add_layer(pool1);
    dnn2.add_layer(conv2);
    dnn2.add_layer(relu_conv2);
    dnn2.add_layer(pool2);
    dnn2.add_layer(fc1);
    dnn2.add_layer(relu_fc1);
    dnn2.add_layer(fc2);
    dnn2.add_layer(relu_fc2);
    dnn2.add_layer(fc3);
    dnn2.add_layer(softmax);

    // loss
    Loss *loss = new CrossEntropy;
    dnn2.add_loss(loss);

    return dnn2;
}