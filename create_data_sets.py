import plotter
import numpy as np
import matrix_io
import random
import sklearn.datasets



# X, y = sklearn.datasets.make_moons(600, noise=0.20)
# plotter.plot_2D_dataset(X, y, "foo.png")
#
# matrix_io.save_dataset("moons", X, y)
#
# X2, y2 = matrix_io.load_dataset("moons")
#
#
# X, y = sklearn.datasets.make_moons(300, noise=0.20)
# plotter.plot_2D_dataset(X, y, "footest.png")
#
# matrix_io.save_dataset("moons_test", X, y)



# from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata('MNIST original')
#
# x = mnist.data
# y = mnist.target
#
# train = []
# test = []
#
# trainy = []
# testy = []
#
# for c in range(70000):
#     if c % 10 == 0:
#         test.append(x[c])
#         testy.append(y[c])
#     else:
#         train.append(x[c])
#         trainy.append(y[c])
#
# train = np.array(train)
# trainy = np.array(trainy)
#
# test = np.array(test)
# testy = np.array(testy)
#
# matrix_io.save_dataset("mnist", train, trainy)
# matrix_io.save_dataset("mnist_test", test, testy)



# X, y = sklearn.datasets.make_blobs(n_samples=600, cluster_std=[1,2,3], shuffle=True)
# plotter.plot_2D_dataset(X, y, "foo.png")
#
# train = []
# test = []
#
# trainy = []
# testy = []
#
# for c in range(400):
#     if c % 2 == 0:
#         test.append(X[c])
#         testy.append(y[c])
#     else:
#         train.append(X[c])
#         trainy.append(y[c])
#
# train = np.array(train)
# trainy = np.array(trainy)
#
# test = np.array(test)
# testy = np.array(testy)
#
# plotter.plot_2D_dataset(test, testy, "foo_test.png")
#
# matrix_io.save_dataset("three_blobs", train, trainy)
# matrix_io.save_dataset("three_blobs_test", test, testy)



X, y = sklearn.datasets.make_blobs(n_samples=600, centers=[[-2,-2],[-2,2],[2,2],[2,-2]], shuffle=True)
plotter.plot_2D_dataset(X, y, "plots/foo.png")

train = []
test = []

trainy = []
testy = []

for c in range(600):

    if y[c] == 2:
        y[c] = 0
    if y[c] == 3:
        y[c] = 1

    if c % 2 == 0:
        test.append(X[c])
        testy.append(y[c])
    else:
        train.append(X[c])
        trainy.append(y[c])

train = np.array(train)
trainy = np.array(trainy)

test = np.array(test)
testy = np.array(testy)

plotter.plot_2D_dataset(test, testy, "plots/foo_test.png")

matrix_io.save_dataset("data/xor", train, trainy)
matrix_io.save_dataset("data/xor_test", test, testy)


