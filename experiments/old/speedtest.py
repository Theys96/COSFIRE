import cosfire as c
import numpy as np
import math as m
import cv2
import sys
from PIL import Image
#import matplotlib.pyplot as plt

numthreads = 4 if (len(sys.argv) < 2) else int(sys.argv[1])
numiterations = 10 if (len(sys.argv) < 3) else int(sys.argv[2])

# Prototype image
proto = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
subject = 1 - np.asarray(Image.open('01_test.tif').convert('RGB'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

timings = {}

for i in range(numiterations):
    cosfire = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (2.4, 1), rhoList=range(0,20,2), sigma0=3,  alpha=0.7,
		rotationInvariance = np.arange(24)/12*np.pi, numthreads = numthreads
	   ).fit(proto, (cx, cy))
    cosfire.transform(subject)
    for t in cosfire.strategy.timings:
        if not(t[0][0] == '\t'):
            if t[0] in timings:
                timings[t[0]].append(t[1])
            else:
                timings[t[0]] = [t[1]]

for step in timings:
    print(step)
    for t in timings[step]:
        print("\t{:7.2f}ms".format(t*1000))

'''
# timings
print("\n --- TIME MEASUREMENTS: Symmetric Filter, {} thread(s) --- ".format(numthreads))
for timing in cosfire_symm.strategy.timings:
	print( "{:7.2f}ms\t{}".format(timing[1]*1000, timing[0]) )
print("\n --- TIME MEASUREMENTS: Asymmetric Filter, {} thread(s) --- ".format(numthreads))
for timing in cosfire_asymm.strategy.timings:
	print( "{:7.2f}ms\t{}".format(timing[1]*1000, timing[0]) )
'''
