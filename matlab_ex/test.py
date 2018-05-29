import cosfire as c
import numpy as np
import math as m
from PIL import Image
#import matplotlib.pyplot as plt

# Prototype image
proto_symm = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
mask = np.asarray(Image.open('mask.tif').convert('L'), dtype=np.float64)
subject = np.asarray(Image.open('input.tif').convert('L'), dtype=np.float64)
(cx, cy) = (100,100)

# Create COSFIRE operator and fit it with the prototype

# Symmetrical filter
cosfire_symm = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (2.4, 1), rhoList=range(0,9,2), sigma0=3,  alpha=0.7,
		rotationInvariance = np.arange(12)/12*np.pi
	   ).fit(proto_symm, (cx, cy))
result_symm = cosfire_symm.transform(subject)

# Asymmetrical filter
cosfire_asymm = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (1.8, 1), rhoList=range(0,23,2), sigma0=2,  alpha=0.1,
		rotationInvariance = np.arange(24)/12*np.pi
	   ).fit(proto_symm, (cx, cy))
# Make asymmetrical
asymmTuples = []
for tupl in cosfire_asymm.strategy.tuples:
	if tupl[1] <= np.pi:
		asymmTuples.append(tupl)
cosfire_asymm.strategy.tuples = asymmTuples
result_asymm = cosfire_asymm.transform(subject)

# Save filtered prototypes for reference
Image.fromarray(c.rescaleImage(cosfire_symm.strategy.protoStack.stack[0].image, 0, 255).astype(np.uint8)).save('filtered_prototype_symm.tif')
Image.fromarray(c.rescaleImage(cosfire_asymm.strategy.protoStack.stack[0].image, 0, 255).astype(np.uint8)).save('filtered_prototype_asymm.tif')

# Add results from symmetrical and asymmetrical operators
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