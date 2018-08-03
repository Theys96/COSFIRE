import variationUtil

'''  --- Strategy B --- '''
variationUtil.setVariation('B')
import cosfire as c
import numpy as np
import time
import cv2

# Prototype image
proto = np.asarray(cv2.imread('ci_scripts/line.png', cv2.IMREAD_GRAYSCALE), dtype=np.float64)
subject = 1 - np.asarray(cv2.imread('ci_scripts/01_test.tif'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

n = 5
stats_threads5 = []
for numthreads in 2**np.array([0,1,2,3,4]):
    for i in range(n):
        t0 = time.time()
        cosfire = c.COSFIRE(
    		c.CircleStrategy(c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,16,2), sigma0=3,  alpha=0.7,
    		rotationInvariance = np.arange(24)/12*np.pi, numthreads = numthreads)
    	   ).fit()
        cosfire.transform(subject)
        for timing in cosfire.strategy.timings:
            if timing[0][0]=='S':
                stats_threads5.append( (numthreads, 1000*timing[1]) )
        print("{}, {}/{}".format(numthreads, i, n))

fh = open('figure28.1b.csv', 'w')
print("threads,ms")
fh.write("threads,ms\n")
for stat in stats_threads5:
    print("{},{}".format(int(stat[0]), int(stat[1])))
    fh.write("{},{}\n".format(int(stat[0]), int(stat[1])))
fh.close
