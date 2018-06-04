import cosfire as c
import numpy as np
import math as m
import cv2
from PIL import Image
import scipy.io as sio

#subject image is -1.1102230246251565e-16 different from matlab
#The filter  produces results with a total error of ~0.07 per row or less than 1e-9 per pixel
#Last step of Rescale and Binnary works as intended

numthreads = 1
def compareMat(mat1,mat2):
	result = True
	if not mat1.shape == mat2.shape:
		print('[ERROR] matrix has different dimentions')
		return False

	diff = np.zeros(mat1.shape[0])
	for x in range(mat1.shape[0]):
		for y in range(mat1.shape[1]):
			if mat1[x,y] - mat2[x,y] > 1e-9:
				result = False
			if not mat1[x,y] == mat2[x,y]:
				diff[x] += abs(mat1[x,y]-mat2[x,y]) #sum the error per row
	#print(diff)
	return result

#load matricies form matlab
mat_contents = sio.loadmat('matlabVariables')
subject_image        = mat_contents['subject_image']    #after div 255, 1 channel, img = 1 - img
matlab_mask          = mat_contents['mask']
response_rot1        = mat_contents['response_rot1']
response_rot2        = mat_contents['response_rot2']
response_imageSum    = mat_contents['response_imageSum']
response_rescaled    = mat_contents['response_rescaled']
output_respimage     = mat_contents['output_respimage']
output_segmented     = mat_contents['output_segmented']

# Prototype image
proto_symm = np.asarray(Image.open('line.png').convert('L'), dtype=np.float64)
mask = np.asarray(Image.open('mask.png').convert('L'), dtype=np.float64)
subject = 255 - np.asarray(Image.open('01_test.tif').convert('RGB'), dtype=np.float64)[:,:,1]
subject = subject/255
(cx, cy) = (100,100)

print('[TEST 1] Prototype center: ', proto_symm[cx,cy])
print('[TEST 2] Mask the same:', compareMat(matlab_mask,mask/255))
print('[TEST 3] Subject the same:', compareMat(subject_image,subject))

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

# filtered prototypes 
#filtered_prototype_symm = c.rescaleImage(cosfire_symm.strategy.protoStack.stack[0].image, 0, 255).astype(np.uint8)
#filtered_prototype_asymm = c.rescaleImage(cosfire_asymm.strategy.protoStack.stack[0].image, 0, 255).astype(np.uint8)

# Add results from symmetrical and asymmetrical operators
result = result_symm + result_asymm

print('[TEST 4] rot1 the same:', compareMat(response_rot1,result_symm))
print('[TEST 5] rot2 the same:', compareMat(response_rot2,result_asymm))
print('[TEST 6] result sum the same:', compareMat(response_imageSum,result))

# Stretching and binarization
result_symm = c.rescaleImage(result_symm, 0, 255)
result_asymm = c.rescaleImage(result_asymm, 0, 255)
result = np.multiply(result, mask)
result = c.rescaleImage(result, 0, 255)
binaryResult = np.where(result > 37, 1, 0)
print('[TEST 7] response the same:', compareMat(response_rescaled,result))
print('[TEST 8] binnary the same:', compareMat(output_segmented,binaryResult))
