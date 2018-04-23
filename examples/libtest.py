import cosfire as c
import numpy as np
import math as m
from PIL import Image
import matplotlib.pyplot as plt


# Prototype image
proto = np.asarray(Image.open('prototype1.png').convert('L'), dtype=np.float64)
(cx, cy) = (50,50)

# Create COSFIRE operator and fit it with the prototype
cosf = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, ([1,2,3], 1), [0,10,20,40]
	   ).fit(proto, (cx, cy))

# Draw filtered prototype
tupleCoords = [( cx+int(round(rho*m.sin(phi))) , cy+int(round(rho*m.cos(phi))) ) for (rho, phi, *_) in cosf.strategy.tuples]
print(cosf.strategy.tuples);
plt.imshow(proto, cmap='gray')
tupleCoords = np.asarray(tupleCoords)
try:
	plt.plot(tupleCoords[:,1], tupleCoords[:,0],"xr")
except:
	print("No tuples found")
plt.show()

