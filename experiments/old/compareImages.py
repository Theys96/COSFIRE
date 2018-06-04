import sys
import numpy as np
from PIL import Image
import cosfire as c

if (len(sys.argv) < 3):
    if (len(sys.argv) == 2):
        name = sys.argv[1]
        img = np.asarray(Image.open(name).convert('L'), dtype=np.float64)
        print(img)
        print(np.max(img))
        print(np.min(img))
        sys.exit()
    else:
        sys.exit("Give two image paths")

name1 = sys.argv[1]
name2 = sys.argv[2]

img1 = np.asarray(Image.open(name1).convert('L'), dtype=np.float64)
img2 = np.asarray(Image.open(name2).convert('L'), dtype=np.float64)

result = img1 - img2

img = Image.fromarray(c.rescaleImage(result, 0, 255).astype(np.uint8))
img.save('comparison.png')

print(result)
print(np.max(result))
print(np.min(result))
print(np.sum(result))
