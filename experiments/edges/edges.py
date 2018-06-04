import cosfire as c
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

proto = np.asarray(Image.open('edge.png').convert('L'), dtype=np.float64)
subject = np.asarray(Image.open('rino.pgm').convert('L'), dtype=np.float64)
(cx, cy) = (50,50)

cosfire = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (1,1), prototype=proto, center=(cx,cy), rhoList=range(0,11,2), sigma0=2,  alpha=0.3,
		rotationInvariance = np.arange(24)/12*np.pi, scaleInvariance=[1]
	   ).fit()
print(cosfire.strategy.tuples)

result = c.rescaleImage(cosfire.transform(subject), 0, 255)
result = 1 - np.where(result > 10, 1, 0)

#plt.imshow(subject, cmap='gray')
#plt.show()
plt.imshow(result, cmap='gray')
plt.show()
img = Image.fromarray(c.rescaleImage(result,0,255).astype(np.uint8))
img.save('result.png')
