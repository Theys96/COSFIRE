import numpy as np
from PIL import Image
import cosfire as c
import cv2
import scipy.signal as signal
import time
#import matplotlib.pyplot as plt

k_matlab = np.loadtxt('csv/k_matlab.csv', delimiter=',')
A_matlab = np.loadtxt('csv/A_matlab.csv', delimiter=',')
B_matlab = np.loadtxt('csv/B_matlab.csv', delimiter=',')

'''
t0 = time.time()
for i in range(0,100):
    B_matlab_python = signal.convolve(A_matlab, np.outer(k_matlab, k_matlab), mode='same')
timing_a = (time.time()-t0)*10
print("signal.convolve: {:6.4f}ms".format( timing_a ))

t0 = time.time()
for i in range(0,100):
    B_matlab_python = cv2.sepFilter2D(A_matlab, -1, k_matlab, k_matlab)
timing_b = (time.time()-t0)*10
print("cv2.sepFilter2D: {:4.2f}ms".format( timing_b ))
'''

#B_matlab_python = cv2.sepFilter2D(A_matlab, -1, k_matlab, k_matlab)
B_matlab_python = signal.convolve(A_matlab, np.outer(k_matlab, k_matlab), mode='same')
B_matlab_python = np.roll(B_matlab_python, -1, axis=0)
B_matlab_python = np.roll(B_matlab_python, -1, axis=1)

#print(np.outer(k_matlab, k_matlab))

print(B_matlab)
print(B_matlab_python)
## cv2.sepFilter2D runs about 23 times as fast as signal.convolve
#print(timing_a/timing_b)

'''
matlab = np.loadtxt('responses_matlab/sigma1.8rho10.csv', delimiter=',')
python = np.loadtxt('responses/sigma1.8rho10.csv', delimiter=',')

size = (matlab.shape[0]*matlab.shape[1])

#print(np.add.reduce(matlab==python, axis=None)/size)

result = python - matlab

print(np.add.reduce(abs(result) > 0.0000001, axis=None)/size)

Image.fromarray(c.rescaleImage(result, 0, 255).astype(np.uint8)).save('compare_csv.png')
'''
