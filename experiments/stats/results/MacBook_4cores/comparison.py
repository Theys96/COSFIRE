import numpy as np
import matplotlib.pyplot as plt

stats = np.loadtxt('comparison.csv', delimiter=',', skiprows=1)

stats1 = stats[stats[:,0]==5]
stats2 = stats[stats[:,0]==9]
stats3 = stats[stats[:,0]==13]
stats4 = stats[stats[:,0]==17]
stats5 = stats[stats[:,0]==21]
print(stats1)

f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.plot(stats5[:,1],stats5[:,2],marker='o')
ax1.plot(stats4[:,1],stats4[:,2],marker='o')
ax1.plot(stats3[:,1],stats3[:,2],marker='o')
ax1.plot(stats2[:,1],stats2[:,2],marker='o')
ax1.plot(stats1[:,1],stats1[:,2],marker='o')
ax1.set_title('Number of threads vs. Computation time of all shift/combine steps\nvariation 5    4 cores')
ax1.axis([1,8,0,3200])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, 'major', 'y')
ax1.legend(('21 tuples','17 tuples','13 tuples','9 tuples','5 tuples'))
#plt.show()
plt.savefig('comparison.pdf')