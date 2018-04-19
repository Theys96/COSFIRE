import cosfire as c
import numpy as np
import math as m
from PIL import Image
import matplotlib.pyplot as plt


# Prototype image
proto = np.asarray(Image.open('tomato.jpg').convert('L'), dtype=np.float64)
(cx, cy) = (305,305)

# Create COSFIRE operator and fit it with the prototype
cosf = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (2.6, 1), [x*5 for x in range(0,60)]
	   ).fit(proto, (cx, cy))

# Draw filtered prototype
tupleCoords = [( cx+int(round(rho*m.sin(phi))) , cy+int(round(rho*m.cos(phi))) ) for (sigma, rho, phi) in cosf.strategy.tuples]
print(tupleCoords);
plt.imshow(cosf.strategy.filteredProto, cmap='gray')
tupleCoords = np.asarray(tupleCoords)
try:
	plt.plot(tupleCoords[:,1], tupleCoords[:,0],"xr")
except:
	print("No tuples found")
plt.show()