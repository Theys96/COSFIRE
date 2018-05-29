from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Function to find maxima in a circular array
# Returns: array of indices
def circularPeaks(array):
    n = len(array)
    d = 0.00005    # Small error correction
    maxima = []
    for i, val in enumerate(array):
        if (array[(i-1)%n]+d < val and array[(i+1)%n]+d < val):
            maxima.append(i)
        elif (abs(array[(i-1)%n] - val) < d and abs(array[(i+1)%n] - val) < d):
            l = r = 0
            k = 1
            while (abs(array[(i-k)%n] - val) < d) and k < n:
                l += 1; k += 1
            if (array[(i-k)%n] > val+d):
                l = 0
            if k == n:
                return maxima
            k = 1
            while (abs(array[(i+k)%n] - val) < d) and k < n:
                r += 1; k += 1
            if (array[(i+k)%n] > val+d):
                r = 0
            if k == n:
                return maxima
            if (l > 0 and r > 0 and (l == r or l + 1 == r)):
                maxima.append(i)
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
    return round(float, 3)

def rescaleImage(image, mn, mx):
    image = normalize(image)*(mx-mn)
    image += mn
    return image

def shiftImage(image, dx, dy):
    '''
    # FOR SHIFTING WITHOUT ROLLING
    shift = image[-dy:,:] if dy <= 0 else image[:-dy,:]
    shift = shift[:,-dx:] if dx <= 0 else shift[:,:-dx]
    pad = np.zeros((np.absolute(dy), shift.shape[1]))
    shift = np.concatenate((shift, pad)) if dy <= 0 else np.concatenate((pad, shift))
    pad = np.zeros((shift.shape[0], np.absolute(dx)))
    shift = np.concatenate((shift, pad), axis=1) if dx <= 0 else np.concatenate((pad, shift), axis=1)
    '''
    shift = np.roll(image, dx, axis=1)
    shift = np.roll(shift, dy, axis=0)
    return shift

def unique(list):
    unique_list = []
    for x in list:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
