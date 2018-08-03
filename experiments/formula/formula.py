import cosfire as c
import numpy as np
import time
from PIL import Image

# Prototype image
proto = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
subject = 1 - np.asarray(Image.open('subject.png').convert('L'), dtype=np.float64)
(cx, cy) = (100,100)

numthreads = 1
vals1 = []
n = 5
x = 24
for i in range(1,x+1):
	val = 0
	for j in range(n):
	    cosfire = c.COSFIRE(
			c.CircleStrategy(c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,16,2), sigma0=3,  alpha=0.7,
			rotationInvariance = np.arange(i)/12*np.pi, numthreads = numthreads)
		   ).fit()
	    cosfire.transform(subject)
	    for timing in cosfire.strategy.timings:
    		if (timing[0][0] == 'S'):
    			val += 1000*timing[1]
	vals1.append( (i, val/n) )
	print("{}/{}".format(i,x))


numthreads = 4
vals4 = []
n = 5
x = 24
for i in range(1,x+1):
	val = 0
	for j in range(n):
	    cosfire = c.COSFIRE(
			c.CircleStrategy(c.DoGFilter, (2.4, 1), prototype=proto, center=(cx,cy), rhoList=range(0,16,2), sigma0=3,  alpha=0.7,
			rotationInvariance = np.arange(i)/12*np.pi, numthreads = numthreads)
		   ).fit()
	    cosfire.transform(subject)
	    for timing in cosfire.strategy.timings:
    		if (timing[0][0] == 'S'):
    			val += 1000*timing[1]
	vals4.append( (i, val/n) )
	print("{}/{}".format(i,x))


'''  --- Plotting --- '''
import matplotlib.pyplot as plt
vals1 = np.array(vals1)
vals4 = np.array(vals4)

f, (ax1) = plt.subplots(1, 1, sharey=True)

ax1.scatter(vals1[:,0],vals1[:,1])
ax1.scatter(vals4[:,0],vals4[:,1])
ax1.set_title('Number of orientations vs. Total computation time\n{} threads'.format(numthreads))
#ax1.axis([0,np.max(stats_threads[:,0])*1.1,0,4000])
ax1.plot([0,vals1[x-1,0]],[0,vals1[x-1,1]])
ax1.plot([0,vals4[x-1,0]],[0,vals4[x-1,1]])

print("1: y = {:7.2f}x ".format(vals1[x-1,1]/vals1[x-1,0]))
print("4: y = {:7.2f}x ".format(vals4[x-1,1]/vals4[x-1,0]))
ax1.set_xlabel('# Orientations')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')

plt.show()