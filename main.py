import plotter
import numpy as np
import matrix_io
import random

import neural_net

def stochastic_epoch(network, X, y):
    order = range(X.shape[0])
    random.shuffle(order)
    for c in order:
        network.backpropagate(X[c], y[c])

def SGD(network, X, y, epochs):
    for c in range(epochs):
        stochastic_epoch(network, X, y)

def train_nn(network, X, y, Xtest, ytest):
    for c in range(10):
        SGD(network, X, y, 50)
        print "training error: " + str(network.classification_error(X, y))
        print "test error    : " + str(network.classification_error(Xtest, ytest))
        plotter.plot_2D_model_predictions(network, X, y, "plots/foo/run_" + str(c) + ".png")



X, y = matrix_io.load_dataset("data/xor")
Xt, yt = matrix_io.load_dataset("data/xor_test")
network = neural_net.NeuralNet([2,4,4,1], 3)
train_nn(network, X, y, Xt, yt)


