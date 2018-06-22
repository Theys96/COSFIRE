import cosfire as c
import numpy as np
import time
import cv2

# Prototype image
proto = np.asarray(cv2.imread('line.png', cv2.IMREAD_GRAYSCALE), dtype=np.float64)
subject = 1 - np.asarray(cv2.imread('01_test.tif'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

cosfire1 = c.COSFIRE(
    c.CircleStrategy(c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,16,2), sigma0=3,  alpha=0.7,
    rotationInvariance = np.arange(24)/12*np.pi, numthreads = 1)
   ).fit()
cosfire1.transform(subject)

cosfire2 = c.COSFIRE(
	c.CircleStrategy(c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,16,2), sigma0=3,  alpha=0.7,
	rotationInvariance = np.arange(24)/12*np.pi, numthreads = 2)
   ).fit()
cosfire2.transform(subject)

print("\n --- TIME MEASUREMENTS: Sequential --- ")
for timing in cosfire1.strategy.timings:
    print( "{:7.2f}ms\t{}".format(timing[1]*1000, timing[0]) )
print("\n --- TIME MEASUREMENTS: 2 thread(s) --- ")
for timing in cosfire2.strategy.timings:
    print( "{:7.2f}ms\t{}".format(timing[1]*1000, timing[0]) )