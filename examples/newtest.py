import cosfire as c
import numpy as np
import math as m
from PIL import Image
import matplotlib.pyplot as plt


# Prototype image
proto1 = np.asarray(Image.open('prototype1.png').convert('L'), dtype=np.float64)
subject = np.asarray(Image.open('01_test_inv.tif').convert('L'), dtype=np.float64)
(cx, cy) = (50,50)

# Create COSFIRE operator and fit it with the prototype
result1 = cosf = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, ([2,3], 1), [0,10,20], 0.2, 2, 0
	   ).fit(proto1, (cx, cy)).transform(subject)

result1 = np.clip(result1, 0, 0.2*result1.max())
plt.imshow(result1 ,cmap='gray')
plt.show()
