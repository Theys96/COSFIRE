import cosfire as c
import numpy as np
import time
from PIL import Image

# Prototype image
proto = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
subject = 1 - np.asarray(Image.open('01_test.tif').convert('RGB'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

stats_tuples = []
tuples_numthreads = 4
for maxRho in range(2,30,2):
    for i in range(3):
        t0 = time.time()
        cosfire = c.COSFIRE(
    		c.CircleStrategy, c.DoGFilter, (2.4, 1), rhoList=range(0,maxRho,2), sigma0=3,  alpha=0.7,
    		rotationInvariance = np.arange(24)/12*np.pi, numthreads = tuples_numthreads
    	   ).fit(proto, (cx, cy))
        cosfire.transform(subject)
        numTuples = len(cosfire.strategy.tuples)
        stats_tuples.append( (numTuples, 1000*(time.time()-t0)) )

stats_threads = []
threads_numtuples = 0
for numthreads in range(1,10):
    for i in range(3):
        t0 = time.time()
        cosfire = c.COSFIRE(
    		c.CircleStrategy, c.DoGFilter, (2.4, 1), rhoList=range(0,16,2), sigma0=3,  alpha=0.7,
    		rotationInvariance = np.arange(24)/12*np.pi, numthreads = numthreads
    	   ).fit(proto, (cx, cy))
        cosfire.transform(subject)
        threads_numtuples = len(cosfire.strategy.tuples)
        stats_threads.append( (numthreads, 1000*(time.time()-t0)) )



'''  --- Plotting --- '''
import matplotlib.pyplot as plt
stats_tuples = np.array(stats_tuples)
stats_threads = np.array(stats_threads)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(stats_tuples[:,0],stats_tuples[:,1])
ax1.set_title('Number of tuples vs. time\n{} threads'.format(tuples_numthreads))
ax1.axis([0,np.max(stats_tuples[:,0])*1.1,0,4000])
ax1.set_xlabel('#Tuples')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')

ax2.scatter(stats_threads[:,0],stats_threads[:,1])
ax2.set_title('Number of threads vs. time\n{} tuples'.format(threads_numtuples))
ax2.axis([0,np.max(stats_threads[:,0])*1.1,0,4000])
ax2.set_xlabel('#Threads')
ax2.set_ylabel('Time (ms)')
ax2.grid(True, 'major', 'y')

plt.show()
