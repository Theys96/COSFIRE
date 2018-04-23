from sklearn.base import BaseEstimator, TransformerMixin
import cosfire
import math as m
import numpy as np

class COSFIRE(BaseEstimator, TransformerMixin):

	def __init__(self, strategy, *pargs, **kwargs):
		self.strategy = strategy(*pargs, **kwargs)

	def fit(self, prototype, *pargs, **kwargs):
		self.strategy.fit(prototype, *pargs, **kwargs)
		return self;

	def transform(self, prototype, *pargs, **kwargs):
		return self.strategy.transform(prototype, *pargs, **kwargs)



class CircleStrategy(BaseEstimator, TransformerMixin):

	def __init__(self, filt, filterArgs, rhoList, T1=0.2, sigma0=0, alpha=0, precision=16):
		self.filterArgs = filterArgs
		self.filt = filt
		self.T1 = T1
		self.rhoList = rhoList
		self.sigma0 = sigma0
		self.alpha = alpha
		self.precision = precision

	def fit(self, prototype, center):
		self.protoStack = cosfire.ImageStack(prototype, self.filt, self.filterArgs, self.T1)
		self.tuples = self.findTuples(self.protoStack, center)

	def transform(self, subject):
		gaus = cosfire.GaussianFilter(self.sigma0)
		images = []
		for tupl in self.tuples:
			rho = tupl[0]
			phi = tupl[1]
			args = tupl[2:]
			dx = int(round(rho*np.cos(phi)))
			dy = int(round(rho*np.sin(phi)))
			if self.alpha != 0:
				gaus = cosfire.GaussianFilter(self.sigma0 + rho*self.alpha)
			images.append(cosfire.shiftImage(self.filt(*args).transform(gaus.transform(subject)), dx, dy).clip(min=0))
		result = np.ones(subject.shape)
		for img in images:
			result = np.multiply(result, img)
		return result

	def findTuples(self, image, center):
		# Init some variables
		(cx, cy) = center
		tuples = []

		# Go over every rho (radius of circles)
		for rho in self.rhoList:
			if rho == 0:
				# Circle with no radius, so just the center point
				val = self.protoStack.valueAtPoint(cx, cy)
				if (val[0] > self.T1):
					tuples.append((rho, 0)+val[1])
			elif rho > 0:
				# Compute points (amount=precision) on the circle of radius rho with center point (cx,cy)
				coords = [ ( cx+int(round(rho*np.cos(phi))) , cy+int(round(rho*np.sin(phi))) )
							for phi in
								[i*np.pi/self.precision*2 for i in range(0,self.precision)]
						 ]
				# Retrieve values on the circle points in the given filtered prototype
				vals = [self.protoStack.valueAtPoint(*coord) for coord in coords]

				# Find peaks in circle
				maxima = cosfire.circularPeaks([x[0] for x in vals])
				tuples.extend([ (rho, i*np.pi/self.precision*2)+vals[i][1] for i in maxima])
		return tuples
