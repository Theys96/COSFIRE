import sys
import numpy as np
from PIL import Image


if (len(sys.argv) < 3):
    sys.exit("Give two image paths")

name1 = sys.argv[1]
name2 = sys.argv[2]

img1 = np.asarray(Image.open(name1).convert('L'), dtype=np.float64)
img2 = np.asarray(Image.open(name2).convert('L'), dtype=np.float64)

result = img1 - img2

print(result)
print(np.max(result))
print(np.min(result))
