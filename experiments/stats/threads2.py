import cosfire as c
import numpy as np
import time
from PIL import Image

# Prototype image
proto = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
subject = 1 - np.asarray(Image.open('01_test.tif').convert('RGB'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

stats_threads = []
threads_numtuples = 0
for numthreads in 2**np.array([0,1,2,3,4]):
    for i in range(10):
        t0 = time.time()
        cosfire = c.COSFIRE(
    		c.CircleStrategy, c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,16,2), sigma0=3,  alpha=0.7,
    		rotationInvariance = np.arange(24)/12*np.pi, numthreads = numthreads
    	   ).fit()
        cosfire.transform(subject)
        threads_numtuples = len(cosfire.strategy.tuples)
        for timing in cosfire.strategy.timings:
            if timing[0][0]=='S':
                stats_threads.append( (numthreads, 1000*timing[1]) )
        print("{}, {}/10".format(numthreads, i))



'''  --- Plotting --- '''
import matplotlib.pyplot as plt
stats_threads = np.array(stats_threads)

f, (ax1) = plt.subplots(1, 1, sharey=True)

for stat in stats_threads:
    print("{} threads,{}ms".format(int(stat[0]), int(stat[1])))

ax1.scatter(stats_threads[:,0],stats_threads[:,1],marker='o')
ax1.set_title('Number of threads vs. Computation time of the shift/blur step\nMultithreading the main loop\n{} tuples'.format(threads_numtuples))
ax1.axis([0,np.max(stats_threads[:,0])*1.1,0,4000])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')

plt.show()
