import cosfire
import numpy

x = numpy.array([
   [1,2,3],
   [2,3,4],
   [4,5,6],
   [1,6,4]])
print(0,0,"\n",cosfire.shiftImage(x, 0, 0))
print(1,1,"\n",cosfire.shiftImage(x, 1, 1))
print(1,2,"\n",cosfire.shiftImage(x, 1, 2))
print(2,0,"\n",cosfire.shiftImage(x, 2, 0))
print(-1,2,"\n",cosfire.shiftImage(x, -1, 2))
print(2,-2,"\n",cosfire.shiftImage(x, 2, -2))
