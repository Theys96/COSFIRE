import cosfire as c
import numpy as np
import math as m
from PIL import Image
import matplotlib.pyplot as plt


# Prototype image
proto = np.asarray(Image.open('prototype3.png').convert('L'), dtype=np.float64)
(cx, cy) = (50,50)

# Create COSFIRE operator and fit it with the prototype
cosf = c.COSFIRE(c.CircleStrategy, c.DoGFilter, (2.6, 1), [0,20,40], 0)
tuples = cosf.fit(proto, (cx, cy))

# Draw filtered prototype
tupleCoords = [( cx+int(round(rho*m.sin(phi))) , cy+int(round(rho*m.cos(phi))) ) for (sigma, rho, phi) in tuples]
plt.imshow(cosf.strategy.filteredProto, cmap='gray')
tupleCoords = np.asarray(tupleCoords)
plt.plot(tupleCoords[:,1], tupleCoords[:,0],"xr")
plt.show()