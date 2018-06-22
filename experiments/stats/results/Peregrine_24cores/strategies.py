import numpy as np
import matplotlib.pyplot as plt

stats1 = np.loadtxt('strategy1.csv', delimiter=',')
stats2 = np.loadtxt('strategy2.csv', delimiter=',')
f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(stats1[:,0],stats1[:,1],marker='o')
ax1.scatter(stats2[:,0],stats2[:,1],marker='o')
ax1.legend(["Old strategy","New strategy"])
ax1.set_title('Number of threads vs. Computation time of all shift/combine steps\n24 cores')
ax1.axis([0,32,0,1500])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
plt.show()
plt.savefig('strategies.pdf')
