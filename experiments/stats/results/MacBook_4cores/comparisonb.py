import numpy as np
import matplotlib.pyplot as plt


stats = np.loadtxt('comparison6b.csv', delimiter=',', skiprows=1)

stats1 = stats[stats[:,0]==1]
stats2 = stats[stats[:,0]==3]
stats3 = stats[stats[:,0]==7]
stats4 = stats[stats[:,0]==15]
stats5 = stats[stats[:,0]==31]
stats6 = stats[stats[:,0]==63]

# Base tuples!!

f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.plot(stats6[:,1],stats6[0,2]/stats6[:,2],marker='o')
ax1.plot(stats5[:,1],stats5[0,2]/stats5[:,2],marker='o')
ax1.plot(stats4[:,1],stats4[0,2]/stats4[:,2],marker='o')
ax1.plot(stats3[:,1],stats3[0,2]/stats3[:,2],marker='o')
ax1.plot(stats2[:,1],stats2[0,2]/stats2[:,2],marker='o')
ax1.plot(stats1[:,1],stats1[0,2]/stats1[:,2],marker='o')
ax1.set_title('Number of threads vs. Speed-up in computation time of all shift/combine steps\nvariation 6    4 cores')
ax1.axis([1,8,0,2])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Relative speedup')
ax1.grid(True, 'major', 'y')
ax1.legend(('63 tuples','31 tuples','15 tuples','7 tuples','3 tuples','1 tuple'))
plt.show()
#plt.savefig('comparison6b.pdf')
