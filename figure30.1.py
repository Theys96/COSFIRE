import os
os.system("python3 figure30.1a.py")
os.system("python3 figure30.1b.py")

'''  --- Plotting --- '''
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

statsX = np.loadtxt('figure30.1a.csv', delimiter=',', skiprows=1)
statsA = np.loadtxt('figure30.1b.csv', delimiter=',', skiprows=1)

f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(statsX[:,0],statsX[:,1],marker='o')
ax1.scatter(statsA[:,0],statsA[:,1],marker='o')
ax1.axis([0, 32, 0, 1300])
ax1.set_title('Number of processes vs. Computation time of a complete subject transformation\n{} core machine'.format(multiprocessing.cpu_count()))
ax1.set_xlabel('Number of processes')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
ax1.legend(["data parallelism", "task parallelism"])
plt.show()
