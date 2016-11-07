import numpy as np

def save_dataset(fname, x, y):
    np.savez(fname, x=x, y=y)

def load_dataset(fname):
    if ".npz" not in fname:
        fname += ".npz"
    npz = np.load(fname)
    return npz['x'], npz['y']