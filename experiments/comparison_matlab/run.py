import cosfire as c
import numpy as np
import math as m
import cv2
from PIL import Image
#import matplotlib.pyplot as plt

numthreads = 4

# Prototype image
proto_symm = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
mask = np.asarray(Image.open('mask.png').convert('L'), dtype=np.float64)
subject = 255 - np.asarray(Image.open('01_test.tif').convert('RGB'), dtype=np.float64)[:,:,1]
subject = subject/255
(cx, cy) = (100,100)

# Save subject for later reference
img = Image.fromarray((subject*255).astype(np.uint8))
img.save('responses/subject.png')

# Symmetrical filter
cosfire_symm = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (2.4, 1), rhoList=range(0,9,2), sigma0=3,  alpha=0.7,
		rotationInvariance = np.arange(12)/12*np.pi, numthreads=numthreads
	   ).fit(proto_symm, (cx, cy))
result_symm = cosfire_symm.transform(subject)

# Asymmetrical filter
cosfire_asymm = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (1.8, 1), rhoList=range(0,23,2), sigma0=2,  alpha=0.1,
		rotationInvariance = np.arange(24)/12*np.pi, numthreads=numthreads
	   ).fit(proto_symm, (cx, cy))
# Make asymmetrical
asymmTuples = []
for tupl in cosfire_asymm.strategy.tuples:
	if tupl[1] <= np.pi:
		asymmTuples.append(tupl)
cosfire_asymm.strategy.tuples = asymmTuples
result_asymm = cosfire_asymm.transform(subject)

# Save filtered prototypes for reference
Image.fromarray(c.rescaleImage(cosfire_symm.strategy.protoStack.stack[0].image, 0, 255).astype(np.uint8)).save('responses/filtered_prototype_symm.tif')
Image.fromarray(c.rescaleImage(cosfire_asymm.strategy.protoStack.stack[0].image, 0, 255).astype(np.uint8)).save('responses/filtered_prototype_asymm.tif')

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
img_symm.save('results/output_symm.png')
img_asymm = Image.fromarray(result_asymm.astype(np.uint8))
img_asymm.save('results/output_asymm.png')
img = Image.fromarray(result.astype(np.uint8))
img.save('results/output.png')
imgBinary = Image.fromarray(binaryResult.astype(np.uint8))
imgBinary.save('results/output_binary.png')

# timings
print("\n --- TIME MEASUREMENTS: Symmetric Filter, {} thread(s) --- ".format(numthreads))
for timing in cosfire_symm.strategy.timings:
	print( "{:7.2f}ms\t{}".format(timing[1]*1000, timing[0]) )
print("\n --- TIME MEASUREMENTS: Asymmetric Filter, {} thread(s) --- ".format(numthreads))
for timing in cosfire_asymm.strategy.timings:
	print( "{:7.2f}ms\t{}".format(timing[1]*1000, timing[0]) )
