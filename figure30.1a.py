import variationUtil

'''  --- Strategy X --- '''
variationUtil.setVariation('X')
import cosfire as c
import numpy as np
import time
import cv2

# Prototype image
proto = np.asarray(cv2.imread('ci_scripts/line.png', cv2.IMREAD_GRAYSCALE), dtype=np.float64)
subject = 1 - np.asarray(cv2.imread('ci_scripts/01_test.tif'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

stats_threads = []
n = 3
grids = {1:(1,1), 2:(1,2), 3:(1,3), 4:(2,2), 6:(2,3), 8:(2,4), 12:(3,4), 16:(4,4), 20:(4,5), 24:(4,6), 30:(5,6)}
for numthreads in [1,2,4,6,8,12,16,20,24,30]:
    for i in range(n):
        cosfire = c.COSFIRE(
    		c.CircleStrategy(c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,16,2), sigma0=3,  alpha=0.7,
    		rotationInvariance = np.arange(24)/12*np.pi, grid= grids[numthreads])
    	   ).fit()
        cosfire.transform(subject)
        for timing in cosfire.strategy.timings:
            if timing[0][0]=='T':
                stats_threads.append( (numthreads, 1000*timing[1]) )
        print("{}, {}/{}".format(numthreads, i, n))

fh = open('figure30.1a.csv', 'w')
print("threads,ms")
fh.write("threads,ms\n")
for stat in stats_threads:
    print("{},{}".format(int(stat[0]), int(stat[1])))
    fh.write("{},{}\n".format(int(stat[0]), int(stat[1])))
fh.close
