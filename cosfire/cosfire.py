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
		self.prototype = prototype
		self.center = center
		self.protoStack = cosfire.ImageStack().push(prototype).applyFilter(self.filt, self.filterArgs)
		self.protoStack.T1 = self.T1
		self.tuples = self.findTuples()

	def transform(self, subject):
		result = self.applyCOSFIRE(subject, self.tuples)
		'''
		for i in range(self.precision):
			self.rotateTuples()
			for factor in [0.2, 0.5, 0.8, 1, 1.5, 2, 3]:
				tuples = [(rho*factor, phi, *params) for (rho, phi, *params) in self.tuples]
				result = np.maximum(result, self.applyCOSFIRE(subject, tuples))
		'''
		return result

	def applyCOSFIRE(self, subject, tuples):
		gaus = cosfire.GaussianFilter(self.sigma0)
		self.responses = []
		for tupl in tuples:
			rho = tupl[0]
			phi = tupl[1]
			args = tupl[2:]
			dx = int(round(rho*np.cos(phi)))
			dy = int(round(rho*np.sin(phi)))
			if self.alpha != 0:
				gaus = cosfire.GaussianFilter(self.sigma0 + rho*self.alpha)
			self.responses.append((cosfire.shiftImage(self.filt(*args).transform(gaus.transform(subject)), -dx, -dy).clip(min=0), rho))

		maxWeight = 2*(np.amax([tupl[0] for tupl in tuples])/3)**2
		totalWeight = 0
		result = np.ones(subject.shape)
		for img in self.responses:
			weight = np.exp(-(img[1]**2)/maxWeight)
			totalWeight += weight
			result = np.multiply(result, img[0]**weight)
		result = result**(1/totalWeight)
		return result

	def findTuples(self):
		# Init some variables
		(cx, cy) = self.center
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
								[i*np.pi/self.precision*2 for i in range(self.precision)]
						 ]
				# Retrieve values on the circle points in the given filtered prototype
				vals = [self.protoStack.valueAtPoint(*coord) for coord in coords]

				# Find peaks in circle
				maxima = cosfire.circularPeaks([x[0] for x in vals])
				tuples.extend([ (rho, i*np.pi/self.precision*2)+vals[i][1] for i in maxima])
		return tuples

	def rotateTuples(self):
		self.tuples = [(rho, phi+(2*np.pi/self.precision), *params) for (rho, phi, *params) in self.tuples]
