import numpy as np
import math

class NeuralNet:

    weights = []
    learning_rate = 0

    # Create fully-connected neural net
    # layer_width_list: number of nodes in each layer, starting with the input and including the output
    #   eg. NeuralNet( [2,5,1] ) operates on size 2x1 inputs, has one hidden layer with 5 nodes, and 1 output node
    # learning_rate: The learning rate for backpropagation- how far we adjust the weights along the gradient
    def __init__(self, layer_width_list, learning_rate):
        self.learning_rate = learning_rate
        self.weights = [np.random.uniform(-3, 3, (layer_width_list[c], layer_width_list[c+1])) for c in range(len(layer_width_list)-1)]
        # defines a vectorized activation function
        # eg. self.activate_layer( [1, 2, 3] ) === [ self.activate(1), self.activate(2), self.activate(3) ]
        self.activate_layer = np.vectorize(self.activate)

    def get_learning_rate(self):
        return self.learning_rate

    def set_learning_rate(self, rate):
        self.learning_rate = rate

    # Single node activation function
    # x: scalar
    def activate(self, x):
        return sigmoid(x)

    # The network's evaluation of a single input
    # x: 1xN vector, where N is the size of the input layer
    # returns: the output layer
    def feed_forward(self, x):
        layer = x
        for W in self.weights:
            layer = self.activate_layer(np.dot(layer, W))
        return layer

    # Run backprop on a single example
    # x: 1xN vector, where N is the size of the input layer
    # y: Integer, the true class of x
    def backpropagate(self, x, y):
        a = [x]
        h = x
        derivatives = []
        for W in self.weights:
            h = self.activate_layer(np.dot(h, W))
            a.append(h)
            derivatives.append(np.zeros(W.shape))


        for layer in range(len(self.weights)-1, -1, -1):
            output = a[layer+1]
            input = a[layer]

            for c in range(len(output)):

                dSigmoid = output[c] * (1 - output[c])

                for i in range(len(input)):
                    partialsum = 0

                    if layer == len(self.weights)-1:
                        #print "output is actual output layer"
                        partialsum = (output[c] - y)
                    else:
                        #print "compute partialsum from upper derivs"
                        for k in range(len(a[layer+2])):
                            partialsum += derivatives[layer + 1][c][k] * self.weights[layer+1][c][k]

                    derivatives[layer][i][c] = dSigmoid * partialsum * input[i]

        #print derivatives

        for layer in range(len(self.weights)):
            self.weights[layer] -= self.learning_rate * derivatives[layer]

    # The network's prediction function for a single input
    # x: Nx1 vector, where N is the size of the input layer
    # returns: the predicted class of x
    def classify(self, x):
        #print self.feed_forward(x)[0]
        return int(round(self.feed_forward(x)[0]))

    def classification_error(self, X, y):
        predictions = [self.classify(x) for x in X]
        mistakes = 0.0
        for c in range(len(y)):
            if predictions[c] != y[c]:
                mistakes += 1
        return mistakes / len(y)



def sigmoid(x):
    return 1 / (1 + math.exp(-x))