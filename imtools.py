import os
from PIL import Image
import numpy as np
from pylab import *

def get_imlist(path):

    return [os.path.join(path, f) for f in os.listdir(path)
                                    if (f.endswith('.jpg') or f.endswith('.bmp'))]

def imresize(im, size):
    """Change the size of array with PIL"""
    pil_im = Image.fromarray(np.array(im).astype(np.uint8))
    return np.array(pil_im.resize(size))

def histeq(im, nbr_bins=256):
    imhist, bins = np.histogram(im, 256, normed=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

def compute_average(imlist):
    """Compute average image for the list"""

    averageim =  array(Image.open(imlist[0]).convert('L'), 'f')

    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname).convert('L'), 'f')
        except:
            print(imname + "... drop")

    averageim /= len(imlist)

    return array(averageim, 'uint8')

def pca(X):
    """Compute eigen vectors, values and mean image of the images array"""
    num_data, dim  = X.shape
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim>num_data:

        M = dot(X, X.T)
        e, EV = np.linalg.eigh(M)
        tmp = dot(X.T, EV).T
        V = tmp[::-1]
        S = np.sqrt(e)[::-1]

        for i in range(V.shape[1]):
            V[:,i] /= S
    else:

        U, S, V = np.linalg.svd(X)
        V = V[:num_data]

    return V, S, mean_X
