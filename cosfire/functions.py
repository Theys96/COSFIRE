from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Function to find maxima in a circular array
# Returns: array of indices
def circularPeaks(array):
    n = len(array)
    up = array[0] > array[n-1]
    maxima = []
    for i, val in enumerate(array):
        #i = i[0]
        added = False
        if i == n-1:
            if up and array[0] < val:
                maxima.append(i)
                added = True
        else:
            if up and (array[i+1] < val):
                maxima.append(i)
                added = True
                up = not up
            elif not up and (array[i+1] > val):
                up = not up
    return maxima

# Set all values < factor*max to 0
def suppress(image, factor):
    maxVal = image.max()
    supImage = np.zeros(shape=image.shape)
    for (x,y), value in np.ndenumerate(image):
        supImage[x,y] = 0 if value < factor*maxVal else value;
    return supImage

def normalize(image):
    image -= image.min()
    return image/image.max()

def shiftImage(image, dx, dy):
    shift = image[-dx:,:] if dx <= 0 else image[:-dx,:]
    shift = shift[:,-dy:] if dy <= 0 else shift[:,:-dy]
    pad = np.zeros((np.absolute(dy), shift.shape[1]))
    shift = np.concatenate((shift, pad)) if dy <= 0 else np.concatenate((pad, shift))
    pad = np.zeros((shift.shape[0], np.absolute(dx)))
    shift = np.concatenate((shift, pad), axis=1) if dx <= 0 else np.concatenate((pad, shift), axis=1)
    return shift