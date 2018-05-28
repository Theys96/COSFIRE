import cosfire as c
import numpy as np
import math as m
from PIL import Image

# Prototype image
proto1 = np.asarray(Image.open('prototype6.png').convert('L'), dtype=np.float64)
mask = np.asarray(Image.open('mask.png').convert('L'), dtype=np.float64)
proto2 = np.asarray(Image.open('prototype5.png').convert('L'), dtype=np.float64)
protoT = np.asarray(Image.open('prototype2.png').convert('L'), dtype=np.float64)
subject = np.asarray(Image.open('01_test_inv.tif').convert('L'), dtype=np.float64)
(cx, cy) = (101,101)

# Create COSFIRE operator and fit it with the prototype
result1 = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (1.8, 1), range(0,20,4), 0.2, 2/6, 0.1/6, precision=36
	   ).fit(proto1, (cx, cy)).transform(subject)
'''
result2 = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (1.8, 1), range(0,20,4), 0.2, 2/6, 0.1/6, precision=36
	   ).fit(proto2, (cx, cy)).transform(subject)
resultT = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (1.8, 1), range(0,20,4), 0.2, 2/6, 0.1/6, precision=36
	   ).fit(protoT, (cx, cy)).transform(subject)
result2 = c.COSFIRE(
		c.CircleStrategy, c.GaborFilter, ([2,4,6,8,10],[0,0.5*np.pi,np.pi,1.5*np.pi],[2,4,6,8,10],1,0), [0, 20, 40]
	   ).fit(proto2, (cx, cy)).transform(subject)
'''


#result = result1 + result2

#result = c.rescaleImage(result, 0, 255)
result1 = c.rescaleImage(np.multiply(result1, mask), 0, 255)

#result2 = c.rescaleImage(result2, 0, 255)
#resultT = c.rescaleImage(resultT, 0, 255)

result = np.where(result1 > 37, 255, 0)

img = Image.fromarray(result.astype(np.uint8))
img.save('output.tif')
img1 = Image.fromarray(result1.astype(np.uint8))
img1.save('output1.tif')
'''
img2 = Image.fromarray(result2.astype(np.uint8))
img2.save('output2.tif')
imgT = Image.fromarray(resultT.astype(np.uint8))
imgT.save('outputT.tif')
'''
