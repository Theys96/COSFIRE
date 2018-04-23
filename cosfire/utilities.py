import cosfire as c
import numpy as np
import math as m
from PIL import Image

class ImageStack():

    def __init__(self, image, filt, filterArgs, T1=0):

        self.T1 = T1

        # Compute all combinations of parameters
        argList = [(v,) for v in filterArgs[0]] if type(filterArgs[0])==list else [(filterArgs[0],)]
        for arg in filterArgs[1:]:
            if type(arg)==list:
                argList = [tupl + (v,) for tupl in argList for v in arg]
            else:
                argList = [tupl + (arg,) for tupl in argList]

        # Apply all possible filters
        self.argList = argList
        self.stack = [(tupl, filt(*tupl).transform(image)) for tupl in self.argList]

    def valueAtPoint(self, x, y):
        val = self.T1
        tupl = ()
        for img in self.stack:
            if img[1][y][x] > val:
                val = img[1][y][x]
                tupl = img[0]
        return val,tupl
