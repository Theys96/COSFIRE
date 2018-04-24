import cosfire
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt

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


# Experimental class
class ImageStack2():

    def __init__():
        stack = []

    def push(image):
        self.stack.append(image)

    def pop():
        return self.stack.pop()

    # Reduce the stack to a single item: the result of a given
    # function after passing it the entire stack as a list
    def join(func, *args):
        stack = [func(self.stack, *args)]

    # Pass all current items in the stack to a given function
    # The function may push new items but these are not passed again later
    def applyAllCurrent(func, *args):
        stack2 = []
        while self.stack:
            func(stack2, self.stack.pop(), *args)
        self.stack = stack2

    # Pass all current items in the stack to a given function
    # The function may push new items and these are passed again later
    # This means this may run indefinitely/infinitely!
    def applyIndef(func, *args):
        while self.stack:
            func(self.stack, self.stack.pop(), *args)

    # Apply a filter to all items in the stack, popping them
    # and pushing the results
    def applyFilter(filt, filterArgs):
        # Compute all combinations of parameters
        argList = [(v,) for v in filterArgs[0]] if type(filterArgs[0])==list else [(filterArgs[0],)]
        for arg in filterArgs[1:]:
            if type(arg)==list:
                argList = [tupl + (v,) for tupl in argList for v in arg]
            else:
                argList = [tupl + (arg,) for tupl in argList]

        # Apply all possible filters
        def apply(stack, item, filt, argList):
            stack.extend([(tupl, filt(*tupl).transform(item)) for tupl in argList])
        self.applyAllCurrent(apply, filt, argList)


'''
proto = np.asarray(Image.open('prototype1.png').convert('L'), dtype=np.float64)
stack = ImageStack(proto, cosfire.DoGFilter, ([1,1.5,2,2.6,3,3.9,5], 1))
for img in stack.stack:
    print(img[0])
    plt.imshow(img[1], cmap='gray')
    plt.show()
'''
