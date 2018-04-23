import cosfire
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

proto = np.asarray(Image.open('prototype1.png').convert('L'), dtype=np.float64)
stack = cosfire.ImageStack(proto, cosfire.DoGFilter, ([1,1.5,2,2.6,3,3.9,5], 1))
for img in stack.stack:
    print(img[0])
    plt.imshow(img[1], cmap='gray')
    plt.show()