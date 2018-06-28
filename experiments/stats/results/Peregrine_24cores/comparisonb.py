import numpy as np
import matplotlib.pyplot as plt


stats = np.loadtxt('comparison6b.csv', delimiter=',', skiprows=1)

stats1 = stats[stats[:,0]==3]
stats2 = stats[stats[:,0]==7]
stats3 = stats[stats[:,0]==15]
stats4 = stats[stats[:,0]==31]
stats5 = stats[stats[:,0]==63]
stats6 = stats[stats[:,0]==127]

# Base tuples!!

f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.plot(stats6[:,1],stats6[0,2]/stats6[:,2],marker='o')
ax1.plot(stats5[:,1],stats5[0,2]/stats5[:,2],marker='o')
ax1.plot(stats4[:,1],stats4[0,2]/stats4[:,2],marker='o')
ax1.plot(stats3[:,1],stats3[0,2]/stats3[:,2],marker='o')
ax1.plot(stats2[:,1],stats2[0,2]/stats2[:,2],marker='o')
ax1.plot(stats1[:,1],stats1[0,2]/stats1[:,2],marker='o')
ax1.set_title('Number of threads vs. Speedup in computation time of all shift/combine steps\nvariation 6    24 cores')
ax1.axis([1,32,0,4])
ax1.set_xlabel('#Threads')
ax1.set_ylabel('Relative speedup')
ax1.grid(True, 'major', 'y')
ax1.legend(('127 tuples','63 tuples','31 tuples','15 tuples','7 tuples','3 tuples'))
plt.show()
#plt.savefig('comparison6b.pdf')
