import variationUtil
variationUtil.setVariation('5')
import cosfire as c

import numpy as np
import time
import cv2
import multiprocessing

# Prototype image
proto = np.asarray(cv2.imread('ci_scripts/line.png', cv2.IMREAD_GRAYSCALE), dtype=np.float64)
subject = 1 - np.asarray(cv2.imread('ci_scripts/01_test.tif'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

stats_threads = {}
n = 5
for maxRho in [4+1,8+1,12+1,16+1,20+1]:
    stats_rho = {}
    for numthreads in 2**np.array([0,1,2,3]):
        stats_rho[numthreads] = 0
        for i in range(n):
            stats_rho[numthreads] += 0
            cosfire = c.COSFIRE(
        		c.CircleStrategy(c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,maxRho,2), sigma0=3,  alpha=0.7,
        		rotationInvariance = np.arange(24)/12*np.pi, numthreads = numthreads)
        	   ).fit()
            cosfire.transform(subject)
            numTuples = len(cosfire.strategy.tuples)
            for timing in cosfire.strategy.timings:
                if timing[0][0]=='S':
                    stats_rho[numthreads] += 1000*timing[1]
            print("{}, {}, {}/{}".format(maxRho, numthreads, i, n))
        stats_rho[numthreads] /= n
    stats_threads[numTuples] = stats_rho

print("tuples,threads,ms")
stats = []
for rho in stats_threads:
    for threads in stats_threads[rho]:
        print("{},{},{}".format(int(rho),int(threads),int(stats_threads[rho][threads])))
        stats.append([int(rho),int(threads),int(stats_threads[rho][threads])])
stats = np.array(stats)

'''  --- Plotting --- '''
import matplotlib.pyplot as plt
stats1 = stats[stats[:,0]==5]
stats2 = stats[stats[:,0]==9]
stats3 = stats[stats[:,0]==13]
stats4 = stats[stats[:,0]==17]
stats5 = stats[stats[:,0]==21]

f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.plot(stats5[:,1],stats5[:,2],marker='o')
ax1.plot(stats4[:,1],stats4[:,2],marker='o')
ax1.plot(stats3[:,1],stats3[:,2],marker='o')
ax1.plot(stats2[:,1],stats2[:,2],marker='o')
ax1.plot(stats1[:,1],stats1[:,2],marker='o')
ax1.set_title('Number of threads vs. Computation time of all shift/combine steps\nvariation 5    {} cores'.format(multiprocessing.cpu_count()))
ax1.axis([1,8,0,3200])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
ax1.legend(('21 tuples','17 tuples','13 tuples','9 tuples','5 tuples'))
plt.show()
