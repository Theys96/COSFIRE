#import matplotlib.pyplot as plt
import numpy as np
#import time
import cosfire as c
import cv2

# Prototype image
proto = np.asarray(cv2.imread('line.png', cv2.IMREAD_GRAYSCALE), dtype=np.float64)
subject = 1 - np.asarray(cv2.imread('01_test.tif'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

stats_threads = {}
n = 3
for maxRho in [10]:
    stats_rho = {}
    for numthreads in 2**np.array([0,1,2,3]):
        stats_rho[numthreads] = 0
        for i in range(n):
            cosfire = c.COSFIRE(
        		c.CircleStrategy(c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,maxRho,1), sigma0=3,  alpha=0.7,
        		rotationInvariance = np.arange(24)/12*np.pi, numthreads = numthreads)
        	   ).fit()
            cosfire.transform(subject)
            numTuples = len(cosfire.strategy.tuples)
            for timing in cosfire.strategy.timings:
                if timing[0][0]=='S':
                    stats_rho[numthreads] += 1000*timing[1]
            print("{}, {}, {}/{}".format(maxRho, numthreads, i, n))
        stats_rho[numthreads] /= 5
    stats_threads[numTuples] = stats_rho

print("tuples,threads,ms")
for rho in stats_threads:
    for threads in stats_threads[rho]:
        print("{},{},{}".format(int(rho),int(threads),int(stats_threads[rho][threads])))
'''
x = np.array([1,2,3])
h = np.array([0.3,1,2])
plt.bar(x,h, width=0.3, color='b')
plt.bar(x-0.3,h, width=0.3, color='r')
plt.show()
'''
