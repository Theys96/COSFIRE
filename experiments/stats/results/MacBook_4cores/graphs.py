import numpy as np
import matplotlib.pyplot as plt

stats = np.loadtxt('results1.csv', delimiter=',', skiprows=1)
f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(stats[:,0],stats[:,1],marker='o')
ax1.set_title('Number of threads vs. Computation time of all shift/combine steps\nvariation 1    4 cores')
ax1.axis([0,np.max(stats[:,0])*1.1,0,2500])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
plt.savefig('results1.pdf')

stats = np.loadtxt('results2.csv', delimiter=',', skiprows=1)
f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(stats[:,0],stats[:,1],marker='o')
ax1.set_title('Number of threads vs. Computation time of all shift/combine steps\nvariation 2    4 cores')
ax1.axis([0,np.max(stats[:,0])*1.1,0,2500])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
plt.savefig('results2.pdf')

stats = np.loadtxt('results3.csv', delimiter=',', skiprows=1)
f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(stats[:,0],stats[:,1],marker='o')
ax1.set_title('Number of threads vs. Computation time of all shift/combine steps\nvariation 3    4 cores')
ax1.axis([0,np.max(stats[:,0])*1.1,0,2500])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
plt.savefig('results3.pdf')

stats = np.loadtxt('results4.csv', delimiter=',', skiprows=1)
f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(stats[:,0],stats[:,1],marker='o')
ax1.set_title('Number of threads vs. Computation time of all shift/combine steps\nvariation 4    4 cores')
ax1.axis([0,np.max(stats[:,0])*1.1,0,2500])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
plt.savefig('results4.pdf')

stats = np.loadtxt('results5.csv', delimiter=',', skiprows=1)
f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.scatter(stats[:,0],stats[:,1],marker='o')
ax1.set_title('Number of threads vs. Computation time of all shift/combine steps\nvariation 5    4 cores')
ax1.axis([0,np.max(stats[:,0])*1.1,0,2500])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
plt.savefig('results5.pdf')