from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Function to find maxima in a circular array
# Returns: array of indices
def circularPeaks(array):
    n = len(array)
    up = array[0] > array[n-1]
    maxima = []
    #print(array)
    for i, val in enumerate(array):
        #i = i[0]
        added = False
        if i == n-1:
            if up and array[0]+0.00005 < val: # +0.00005 Required for precision errors
                maxima.append(i)
                added = True
        else:
            if up and (array[i+1]+0.00005 <= val): # +0.00005 Required for precision errors
                maxima.append(i)
                added = True
                up = not up
            elif not up and (array[i+1] > val+0.00005):
                up = not up
    #print(maxima)
    return maxima

# Set all values < factor*max to 0
def suppress(image, factor):
    maxVal = image.max()
    supImage = np.zeros(shape=image.shape)
    for (x,y), value in np.ndenumerate(image):
        supImage[x,y] = 0 if value < factor*maxVal else value;
    return supImage

def normalize(image):
    mn = image.min()
    mx = image.max()
    image -= mn
    return image/(mx-mn)

def approx(float):
    return round(float, 5)

def rescaleImage(image, mn, mx):
    image = normalize(image)*(mx-mn)
    image += mn
    return image

def shiftImage(image, dx, dy):
    shift = image[-dy:,:] if dy <= 0 else image[:-dy,:]
    shift = shift[:,-dx:] if dx <= 0 else shift[:,:-dx]
    pad = np.zeros((np.absolute(dy), shift.shape[1]))
    shift = np.concatenate((shift, pad)) if dy <= 0 else np.concatenate((pad, shift))
    pad = np.zeros((shift.shape[0], np.absolute(dx)))
    shift = np.concatenate((shift, pad), axis=1) if dx <= 0 else np.concatenate((pad, shift), axis=1)
    '''
    shift = np.roll(image, dx, axis=1)
    shift = np.roll(shift, dy, axis=0)
    '''
    return shift
