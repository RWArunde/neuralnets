import matplotlib
import numpy as np
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def plot_2D_model_predictions(model, X, y, fname):
    fig, ax = plt.subplots()

    xmin, xmax = X[:, 0].min() - .5, X[:, 0].max() + .5
    ymin, ymax = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
    Z = np.array( [ model.classify(x) for x in (np.c_[xx.ravel(), yy.ravel()]) ] )
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

    fig.savefig(fname)
    plt.close(fig)

def plot_2D_dataset(X, y, fname):
    fig, ax = plt.subplots()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    fig.savefig(fname)
    plt.close(fig)

