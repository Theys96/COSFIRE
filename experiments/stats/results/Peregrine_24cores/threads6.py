import numpy as np
import matplotlib.pyplot as plt


stats = np.loadtxt('threads6.csv', delimiter=',', skiprows=1)

stats1 = stats[stats[:,1]==8]
stats2 = stats[stats[:,1]==24]

# Base tuples!!

f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.plot(stats1[:,0],stats1[:,2],marker='o')
ax1.plot(stats2[:,0],stats2[:,2],marker='o')
ax1.set_title('Number of tuples vs. Speedup in computation time of all shift/combine steps\nvariation 6    24 cores')
ax1.axis([0,160,1,4])
ax1.set_xlabel('Number of tuples')
ax1.set_ylabel('Relative speedup')
ax1.grid(True, 'major', 'y')
ax1.legend(('8 threads','24 threads'))
plt.show()
#plt.savefig('comparison6b.pdf')
