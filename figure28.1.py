import os
os.system("python3 figure28.1a.py")
os.system("python3 figure28.1b.py")
os.system("python3 figure28.1c.py")

'''  --- Plotting --- '''
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

resultA = np.loadtxt('figure28.1a.csv', delimiter=',', skiprows=1)
resultB = np.loadtxt('figure28.1b.csv', delimiter=',', skiprows=1)
result5 = np.loadtxt('figure28.1c.csv', delimiter=',', skiprows=1)

f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(result5[:,0],result5[:,1],marker='x')
ax1.scatter(resultB[:,0],resultB[:,1],marker='x')
ax1.scatter(resultA[:,0],resultA[:,1],marker='x')
ax1.legend(["Variation 5", "Strategy B/Varation 6", "Strategy A"])
ax1.set_title('Number of threads vs. Computation time of all shift/combine steps\n{} cores'.format(multiprocessing.cpu_count()))
ax1.axis([0,10,0,3000])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
#plt.savefig('graphAB5.pdf')
plt.show()
