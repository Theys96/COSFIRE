import cosfire as c
import numpy as np
import math as m
from PIL import Image
#import matplotlib.pyplot as plt

# Prototype image
proto_symm = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
mask = np.asarray(Image.open('mask.png').convert('L'), dtype=np.float64)
proto_asymm = np.asarray(Image.open('line_ending.png').convert('L'), dtype=np.float64)
subject = np.asarray(Image.open('01_test_inv.tif').convert('L'), dtype=np.float64)
(cx, cy) = (100,100)

# Create COSFIRE operator and fit it with the prototype
print("ROT1 starts here! @@@@@@")
cosfire_symm = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (2.4, 1), rhoList=range(0,9,2), sigma0=3/6,  alpha=0.7/6
	   ).fit(proto_symm, (cx, cy))
result_symm = cosfire_symm.transform(subject)

print("ROT2 starts here! @@@@@@")
cosfire_asymm = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (1.8, 1), rhoList=range(0,23,2), sigma0=2/6,  alpha=0.1/6
	   ).fit(proto_symm, (cx, cy))

# Make asymmetrical
asymmTuples = []
for tupl in cosfire_asymm.strategy.tuples:
	if tupl[1] <= np.pi:
		asymmTuples.append(tupl)
cosfire_asymm.strategy.tuples = asymmTuples

result_asymm = cosfire_asymm.transform(subject)
print("ROT2 ends here! @@@@@@")

Image.fromarray(c.rescaleImage(cosfire_symm.strategy.protoStack.stack[0].image, 0, 255).astype(np.uint8)).save('filtered_prototype_symm.tif')
Image.fromarray(c.rescaleImage(cosfire_asymm.strategy.protoStack.stack[0].image, 0, 255).astype(np.uint8)).save('filtered_prototype_asymm.tif')

result = result_symm + result_asymm

# Stretching and binarization
result_symm = c.rescaleImage(result_symm, 0, 255)
result_asymm = c.rescaleImage(result_asymm, 0, 255)

result = np.multiply(result, mask)
result = c.rescaleImage(result, 0, 255)
binaryResult = np.where(result > 37, 255, 0)

# Saving
img_symm = Image.fromarray(result_symm.astype(np.uint8))
img_symm.save('output_symm.tif')
img_asymm = Image.fromarray(result_asymm.astype(np.uint8))
img_asymm.save('output_asymm.tif')
img = Image.fromarray(result.astype(np.uint8))
img.save('output.tif')
imgBinary = Image.fromarray(binaryResult.astype(np.uint8))
imgBinary.save('output_binary.tif')
