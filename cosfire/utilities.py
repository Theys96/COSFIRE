import cosfire as c
import numpy as np
import math as m
from PIL import Image
import matplotlib.pyplot as plt

class ImageStack():

    def __init__(self, image, filt, filterArgs):

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
