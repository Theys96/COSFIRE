import cosfire as c
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

data = np.loadtxt('input.csv', delimiter=',')
print(data)

tresh = 0.5
masked = cv2.imread('masked.png', cv2.IMREAD_GRAYSCALE)
L = cv2.imread('01_test.tif', cv2.IMREAD_GRAYSCALE)/100.0
mask = 1 - (L < tresh)

plt.imshow(data, cmap='gray')
plt.show()
