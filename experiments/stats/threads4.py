import cosfire as c
import numpy as np
import time
from PIL import Image

# Prototype image
proto = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
subject = 1 - np.asarray(Image.open('01_test.tif').convert('RGB'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

stats_strategy1 = []
stats_strategy2 = []
threads_numtuples = 0
n = 5
for numthreads in 2**np.array([0,1,2,3,4]):
    for i in range(n):
        cosfire = c.COSFIRE(
    		c.CircleStrategy(c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,16,2), sigma0=3,  alpha=0.7,
    		rotationInvariance = np.arange(24)/12*np.pi, numthreads = numthreads)
    	   ).fit()
        cosfire.transform(subject)
        numtuples = len(cosfire.strategy.tuples)
        for timing in cosfire.strategy.timings:
            if (timing[0] == "Shifting and combining all responses"):
                stats_strategy1.append( (numthreads, 1000*timing[1]) )
            elif (timing[0] == "Shifting and combining all responses 2"):
                stats_strategy2.append( (numthreads, 1000*timing[1]) )
        print("{}, {}/{}".format(numthreads, i, n))



'''  --- Plotting --- '''
import matplotlib.pyplot as plt
stats_strategy1 = np.array(stats_strategy1)
stats_strategy2 = np.array(stats_strategy2)

f, (ax1) = plt.subplots(1, 1, sharey=True)

ax1.scatter(stats_strategy1[:,0],stats_strategy1[:,1])
ax1.scatter(stats_strategy2[:,0],stats_strategy2[:,1])
ax1.legend(["Old strategy","New strategy"])
ax1.set_title('Number of threads vs. Total computation time\n{} tuples'.format(numtuples))
ax1.axis([0,np.max(stats_strategy1[:,0])*1.1,0,4000])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')

plt.show()
