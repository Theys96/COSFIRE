import cosfire as c
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

proto = np.asarray(Image.open('edge.png').convert('L'), dtype=np.float64)
subject = np.asarray(Image.open('road.jpg').convert('L'), dtype=np.float64)
(cx, cy) = (51,50)

'''
cosfire = c.COSFIRE(
<<<<<<< HEAD
		c.CircleStrategy, c.DoGFilter, (2.4,[0,1]), prototype=proto, center=(cx,cy), rhoList=[0,2,4,6,8], sigma0=2.4,  alpha=0.1,
		rotationInvariance = np.arange(24)/12*np.pi, scaleInvariance=[1]
	   ).fit()
'''
cosfire = c.COSFIRE(
		c.CircleStrategy, c.DoGFilter, (2.4,[0,1]), prototype=proto, center=(cx,cy), rhoList=range(0,11,2), sigma0=3,  alpha=0.4,
		rotationInvariance = np.arange(24)/12*np.pi, scaleInvariance=[1]
=======
		c.CircleStrategy("DoGFilter", (1,1), prototype=proto, center=(cx,cy), rhoList=range(0,11,2), sigma0=2,  alpha=0.3,
		rotationInvariance = np.arange(24)/12*np.pi, scaleInvariance=[1])
>>>>>>> e110ad8cdb9b23999853b15ec4f370a2ef50be53
	   ).fit()
print(cosfire.strategy.tuples)

result = c.rescaleImage(cosfire.transform(subject), 0, 255)
result = np.where(result > 0.7*255, 0.7*255, result)
result = c.rescaleImage(result, 0, 255)

#plt.imshow(subject, cmap='gray')
#plt.show()
plt.imshow(result, cmap='gray')
plt.show()
img = Image.fromarray(c.rescaleImage(result,0,255).astype(np.uint8))
img.save('result.png')
