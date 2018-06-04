import matplotlib.pyplot as plt
import numpy as np
#import time
from PIL import Image
import cosfire as c

# Prototype image
proto = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
subject = 1 - np.asarray(Image.open('01_test.tif').convert('RGB'), dtype=np.float64)[:,:,1]
(cx, cy) = (100,100)

stats_threads = {}
for numthreads in 2**np.array([0,1,2]):
    stats_threads[numthreads] = 0
    for i in range(3):
        stats_threads[numthreads] += 0
        cosfire = c.COSFIRE(
    		c.CircleStrategy, c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,16,2), sigma0=3,  alpha=0.7,
    		rotationInvariance = np.arange(24)/12*np.pi, numthreads = numthreads
    	   ).fit()
        cosfire.transform(subject)
        for timing in cosfire.strategy.timings:
            if timing[0][0]=='S':
                stats_threads[numthreads] += 1000*timing[1]
        print("{}, {}/10".format(numthreads, i))
    stats_threads[numthreads] /= 5

print(stats_threads)
'''
x = np.array([1,2,3])
h = np.array([0.3,1,2])
plt.bar(x,h, width=0.3, color='b')
plt.bar(x-0.3,h, width=0.3, color='r')
plt.show()
'''
