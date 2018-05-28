import cosfire as c
import numpy as np
import math as m
from PIL import Image

# Prototype image
proto_symm = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
subject = np.asarray(Image.open('01_test_inv.tif').convert('L'), dtype=np.float64)
(cx, cy) = (101,101)

# Create COSFIRE operator and fit it with the prototype
result = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (1.8, 1), range(0,8,2), 0.2, 3/6, 0.7/6, precision=12
	   ).fit(proto_symm, (cx, cy)).transform(subject)

# Stretching and binarization
result = c.rescaleImage(result, 0, 255)
binaryResult = np.where(result > 255*0.1, 255, 0)

img = Image.fromarray(result.astype(np.uint8))
img.save('output.tif')
imgBinary = Image.fromarray(binaryResult.astype(np.uint8))
imgBinary.save('output_binary.tif')
