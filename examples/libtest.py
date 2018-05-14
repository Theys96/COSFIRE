import cosfire as c
import numpy as np
import math as m
from PIL import Image
import matplotlib.pyplot as plt


# Prototype image
proto = np.asarray(Image.open('prototype1.png').convert('L'), dtype=np.float64)
subject = np.asarray(Image.open('01_test_inv.tif').convert('L'), dtype=np.float64)
(cx, cy) = (50,50)

# Create COSFIRE operator and fit it with the prototype
cosf = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, ([1,2,3,4], 1), [0,10,20], 0.2, 2, 0.
	   ).fit(proto, (cx, cy))
'''
cosf = c.COSFIRE(
		c.CircleStrategy, c.GaborFilter, ([1,2,3,4],0,[1,2],1,0), [0, 20, 40]
	   ).fit(proto, (cx, cy))
'''

for tupl in cosf.strategy.tuples:
	print(tupl)

plt.imshow(cosf.strategy.protoStack.stack[0].image, cmap='gray')
tupleCoords = [( cy+int(round(rho*m.sin(phi))) , cx+int(round(rho*m.cos(phi))) ) for (rho, phi, *_) in cosf.strategy.tuples]
tupleCoords = np.asarray(tupleCoords)
try:
	plt.plot(tupleCoords[:,1], tupleCoords[:,0],"xr")
except:
	print("No tuples found")
plt.show()

result = cosf.transform(subject)

'''
for img in cosf.strategy.responses:
	plt.imshow(img[0], cmap='gray')
	plt.show()
'''

'''
# Draw filtered prototype
tupleCoords = [( cx+int(round(rho*m.sin(phi))) , cy+int(round(rho*m.cos(phi))) ) for (rho, phi, *_) in cosf.strategy.tuples]
print(cosf.strategy.tuples);
#plt.imshow(proto, cmap='gray')
plt.imshow(cosf.strategy.protoStack.stack[0][1], cmap='gray')
tupleCoords = np.asarray(tupleCoords)
try:
	plt.plot(tupleCoords[:,1], tupleCoords[:,0],"xr")
except:
	print("No tuples found")
plt.show()
'''

plt.imshow(result, cmap='gray')
plt.show()
