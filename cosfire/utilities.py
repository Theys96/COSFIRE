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


class ImageObject():

    def __init__(self, image, *args, **kwargs):
        self.image = image
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)


# Experimental class
class ImageStack2():

    def __init__(self):
        self.stack = []
        self.T1 = 0

    def push(self, image):
        if type(image) is ImageObject:
            self.stack.append(image)
        else:
            self.stack.append(ImageObject(image))
        return self

    def pop(self):
        return self.stack.pop()

    # Reduce the stack to a single item: the result of a given
    # function after passing it the entire stack as a list
    def join(self, func, *args):
        self.stack = [func(self.stack, *args)]
        return self

    # Pass all current items in the stack to a given function
    # The function may push new items but these are not passed again later
    def applyAllCurrent(self, func, *args):
        stack2 = []
        while self.stack:
            func(stack2, self.stack.pop(), *args)
        self.stack = stack2
        return self

    # Pass all current items in the stack to a given function
    # The function may push new items and these are passed again later
    # This means this may run indefinitely/infinitely!
    def applyIndef(self, func, *args):
        while self.stack:
            func(self.stack, self.stack.pop(), *args)
        return self

    # Apply a filter to all items in the stack, popping them
    # and pushing the results
    def applyFilter(self, filt, filterArgs):
        # Compute all combinations of parameters
        argList = [(v,) for v in filterArgs[0]] if type(filterArgs[0])==list else [(filterArgs[0],)]
        for arg in filterArgs[1:]:
            if type(arg)==list:
                argList = [tupl + (v,) for tupl in argList for v in arg]
            else:
                argList = [tupl + (arg,) for tupl in argList]

        # Apply all possible filters
        def apply(stack, item, filt, argList):
            stack.extend([ImageObject(filt(*tupl).transform(item.image), params=tupl) for tupl in argList])
        self.applyAllCurrent(apply, filt, argList)

        return self

    def valueAtPoint(self, x, y):
        val = self.T1
        params = None
        for img in self.stack:
            if img.image[y][x] > val:
                val = img.image[y][x]
                params = img.params
        return val,params

