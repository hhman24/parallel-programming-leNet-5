#include "dnnNetwork.h"

Network dnnNetwork_CPU()
{
    Network dnn;

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
    dnn.add_layer(conv1);
    dnn.add_layer(relu_conv1);
    dnn.add_layer(pool1);
    dnn.add_layer(conv2);
    dnn.add_layer(relu_conv2);
    dnn.add_layer(pool2);
    dnn.add_layer(fc1);
    dnn.add_layer(relu_fc1);
    dnn.add_layer(fc2);
    dnn.add_layer(relu_fc2);
    dnn.add_layer(fc3);
    dnn.add_layer(softmax);

    // loss
    Loss *loss = new CrossEntropy;
    dnn.add_loss(loss);

    // load weitghts

    return dnn;
}

Network dnnNetwork_GPU()
{
    Network dnn;

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
    dnn.add_layer(conv1);
    dnn.add_layer(relu_conv1);
    dnn.add_layer(pool1);
    dnn.add_layer(conv2);
    dnn.add_layer(relu_conv2);
    dnn.add_layer(pool2);
    dnn.add_layer(fc1);
    dnn.add_layer(relu_fc1);
    dnn.add_layer(fc2);
    dnn.add_layer(relu_fc2);
    dnn.add_layer(fc3);
    dnn.add_layer(softmax);

    // loss
    Loss *loss = new CrossEntropy;
    dnn.add_loss(loss);

    // load weitghts

    return dnn;
}