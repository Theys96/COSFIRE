import cosfire as c
import numpy as np
import math as m
from PIL import Image

# Prototype image
proto = np.asarray(Image.open('prototype1.png').convert('L'), dtype=np.float64)
subject = np.asarray(Image.open('01_test_inv.tif').convert('L'), dtype=np.float64)
(cx, cy) = (50,50)

# Create COSFIRE operator and fit it with the prototype
result = cosf = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (1.8, 1), range(0,20,4), 0.2, 2/6, 0.1/6, precision=60
	   ).fit(proto, (cx, cy)).transform(subject)
'''
result = c.COSFIRE(
		c.CircleStrategy, c.GaborFilter, ([2,4,6,8,10],[0,0.5*np.pi,np.pi,1.5*np.pi],[2,4,6,8,10],1,0), [0, 20, 40]
	   ).fit(proto, (cx, cy)).transform(subject)
'''

result *= 0.5
result = c.rescaleImage(result, 0, 255)
img = Image.fromarray(result.astype(np.uint8))
img.save('output.tif')
