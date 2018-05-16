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
result = cosf = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, ([2,3], 1), [0,10,20], 0.2, 2, 0
	   ).fit(proto, (cx, cy)).transform(subject)
'''
result = c.COSFIRE(
		c.CircleStrategy, c.GaborFilter, ([2,4,6,8,10],[0,0.5*np.pi,np.pi,1.5*np.pi],[2,4,6,8,10],1,0), [0, 20, 40]
	   ).fit(proto, (cx, cy)).transform(subject)
'''

result = np.clip(result, 0, 0.2*result.max())
plt.imshow(result ,cmap='gray')
plt.show()
