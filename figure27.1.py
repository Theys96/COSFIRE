import os
os.system("python3 figure27.1a.py")
os.system("python3 figure27.1b.py")

'''  --- Plotting --- '''
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

stats5 = np.loadtxt('figure27.1a.csv', delimiter=',', skiprows=1)
stats6 = np.loadtxt('figure27.1b.csv', delimiter=',', skiprows=1)

f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(stats5[:,0],stats5[:,1],marker='o')
ax1.scatter(stats6[:,0],stats6[:,1],marker='o')
ax1.legend(["variation 5","variation 6"])
ax1.set_title('Number of threads vs. Computation time of all shift/combine steps\n{} cores'.format(multiprocessing.cpu_count()))
ax1.axis([0,np.max(stats6[:,0])*1.1,0,2500])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
plt.show()
