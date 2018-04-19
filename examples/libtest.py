import cosfire as c
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Prototype image
proto = np.asarray(Image.open('prototype3.png').convert('L'), dtype=np.float64)

# Create COSFIRE operator and fit it with the prototype
cosf = c.COSFIRE(c.CircleStrategy, c.DoGFilter, (2.6, 1), 0)
protoDoG = cosf.fit(proto)

# Draw filtered prototype
plt.imshow(protoDoG, cmap='gray')
plt.show()