import numpy as np
import matplotlib.pyplot as plt

resultA = np.loadtxt('A.csv', delimiter=',', skiprows=1)
resultB = np.loadtxt('B.csv', delimiter=',', skiprows=1)
result5 = np.loadtxt('5.csv', delimiter=',', skiprows=1)
f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(result5[:,0],result5[:,1],marker='x')
ax1.scatter(resultB[:,0],resultB[:,1],marker='x')
ax1.scatter(resultA[:,0],resultA[:,1],marker='x')
ax1.legend(["Variation 5", "Strategy B/Varation 6", "Strategy A"])
ax1.set_title('Number of threads vs. Computation time of all shift/combine steps\n4 cores')
ax1.axis([0,10,0,3000])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
#plt.savefig('graphAB5.pdf')
plt.show()

