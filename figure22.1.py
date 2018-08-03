import variationUtil
variationUtil.setVariation('1')
import cosfire as c

import numpy as np
import time
import cv2
import multiprocessing

# Prototype image
proto = np.asarray(cv2.imread('ci_scripts/line.png', cv2.IMREAD_GRAYSCALE), dtype=np.float64)
subject = 1 - np.asarray(cv2.imread('ci_scripts/01_test.tif'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

stats_threads = []
n = 5
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
                stats_threads.append( (numthreads, 1000*timing[1]) )
        print("{}, {}/{}".format(numthreads, i, n))

print("threads,ms")
for stat in stats_threads:
    print("{},{}".format(int(stat[0]), int(stat[1])))

'''  --- Plotting --- '''
import matplotlib.pyplot as plt
stats_threads = np.array(stats_threads)

f, (ax1) = plt.subplots(1, 1, sharey=True)

ax1.scatter(stats_threads[:,0],stats_threads[:,1],marker='o')
ax1.set_title('Number of threads vs. Computation time of all shift/blur steps\nvariation 1   {} cores'.format(multiprocessing.cpu_count()))
ax1.axis([0,np.max(stats_threads[:,0])*1.1,0,2500])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')

plt.show()
