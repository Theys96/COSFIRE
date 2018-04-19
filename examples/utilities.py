import cosfire as c
import numpy as np
import math as m
from PIL import Image
import matplotlib.pyplot as plt

class ImageStack():

    def __init__(self, image, filt, filterArgs):

        # Compute all combinations of parameters
        if type(filterArgs[0])==list:
            argList = [(v,) for v in filterArgs[0]]
        else:
            argList = [(filterArgs[0],)]

        for arg in filterArgs[1:]:
            if type(arg)==list:
                argList = [tupl + (v,) for tupl in argList for v in arg]
            else:
                argList = [tupl + (arg,) for tupl in argList]

        # Apply all possible filters
        self.argList = argList
        self.stack = [(tupl, filt(*tupl).transform(image)) for tupl in self.argList]


proto = np.asarray(Image.open('prototype1.png').convert('L'), dtype=np.float64)
stack = ImageStack(proto, c.DoGFilter, ([1,1.5,2,2.6,3,3.9,5], 1))
for img in stack.stack:
    print(img[0])
    plt.imshow(img[1], cmap='gray')
    plt.show()