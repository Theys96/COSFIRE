from sklearn.base import BaseEstimator, TransformerMixin
import math as m
import cv2
import scipy.signal as signal
from .base import FunctionFilter
import numpy as np

class GaussianFilter(FunctionFilter):
    def __init__(self, sigma, sz=-1):
        sz = sigma2sz(sigma) if sz < 0 else sz
        kernel = cv2.getGaussianKernel(sz, sigma)
        if not(sz < 0):
            print(sigma, kernel)
        super().__init__(_sepFilter2D, kernel)

class DoGFilter(FunctionFilter):
    def __init__(self, sigma, onoff, sigmaRatio=0.5):
        sz = sigma2sz(sigma)
        kernel1 = np.outer(cv2.getGaussianKernel(sz, sigma),cv2.getGaussianKernel(sz, sigma))
        kernel2 = np.outer(cv2.getGaussianKernel(sz, sigma*sigmaRatio),cv2.getGaussianKernel(sz, sigma*sigmaRatio))
        if (onoff):
            kernel = kernel2 - kernel1
        else:
            kernel = kernel1 - kernel2
        super().__init__(_Filter2D, kernel)

class GaborFilter(FunctionFilter):
    def __init__(self, sigma, theta, lambd, gamma, psi):
        sz = sigma2sz(sigma)
        kernel = cv2.getGaborKernel((sz, sz), sigma, theta, lambd, gamma, psi);
        super().__init__(_Filter2D, kernel);

class CLAHE(FunctionFilter):
    def __init__(self):
        clahe = cv2.createCLAHE(tileGridSize=(8, 8), clipLimit=0.01, distribution='uniform', alpha=0.4)
        super().__init__(_CLAHE, clahe);

# Executes a 2D convolution by using a 1D kernel twice
def _sepFilter2D(image, kernel):
    return cv2.sepFilter2D(image, -1, kernel, kernel)

# Executes a 2D convolution by using a 2D kernel
def _Filter2D(image, kernel):
    '''
    padX = int((kernel.shape[0]-1)/2)
    padY = int((kernel.shape[1]-1)/2)
    result = signal.convolve(image, kernel, mode='valid')
    result = np.pad(result, ((padY, padY), (padX, padX)), 'constant', constant_values=((0,0),(0,0)))
    '''
    result = signal.convolve(image, kernel, mode='same')
    return result

# Executes Contrast Limited Adaptive Histogram Equalization
def _CLAHE(image, clahe):
    return clahe.apply(image)

def sigma2sz(sigma):
    return m.ceil(sigma*3)*2 + 1; # Guaranteed to be odd
