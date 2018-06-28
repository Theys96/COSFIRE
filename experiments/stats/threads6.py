import cosfire as c
import numpy as np
import time
from PIL import Image

# Prototype image
proto = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
subject = 1 - np.asarray(Image.open('01_test.tif').convert('RGB'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

numthreads = 4

cosfire = c.COSFIRE(
	c.CircleStrategy(c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,16,2), sigma0=3,  alpha=0.7,
	rotationInvariance = np.arange(24)/12*np.pi, numthreads = numthreads)
   ).fit()

print("numthreads={}".format(numthreads))

for chunkSize in range(30):
    t0 = time.time()
    cosfire.chunkSize = chunkSize
    cosfire.transform(subject)
    print("chunksize={}, t={:1.4f}".format(chunkSize, time.time()-t0))



'''  --- Plotting ---
import matplotlib.pyplot as plt
stats_threads = np.array(stats_threads)

f, (ax1) = plt.subplots(1, 1, sharey=True)

ax1.scatter(stats_threads[:,0],stats_threads[:,1])
ax1.set_title('Number of threads vs. Total computation time\n{} tuples'.format(threads_numtuples))
ax1.axis([0,np.max(stats_threads[:,0])*1.1,0,4000])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')

plt.show()
'''
