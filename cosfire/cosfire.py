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
		self.deltaRhoList = [0.2, 0.5, 0.8, 1, 1.5, 2, 3]

	def fit(self, prototype, center):
		self.prototype = prototype
		self.center = center
		self.protoStack = cosfire.ImageStack().push(prototype).applyFilter(self.filt, self.filterArgs)
		self.protoStack.T1 = self.T1
		self.tuples = self.findTuples()

	def transform(self, subject):

		# Precompute all blurred filter responses
		responses = self.computeResponses(subject)

		# Shifting and putting them together
		result = np.zeros(subject.shape)
		for deltaPhi in range(self.precision):   # rotation invariance
			for deltaRho in self.deltaRhoList:   # scale invariance
				# Adjust base tuples
				curTuples = self.computeTuples(deltaPhi, deltaRho)

				# Compute shifted filter responses
				curResponses = []
				for tupl in curTuples:
					rho = tupl[0]
					phi = tupl[1]
					args = tupl[2:]
					sigma = self.sigma0 + rho*self.alpha
					dx = int(round(rho*np.cos(phi)))
					dy = int(round(rho*np.sin(phi)))
					curResponses.append( (cosfire.shiftImage(responses[(sigma,)+tupl[2:]], -dx, -dy).clip(min=0), rho) )

				# Combine shifted filter responses
				curResult = self.weightedGeometricMean(curResponses)

				# Include it in the final result
				result = np.maximum(result, curResult)

		return result

		'''
		result = self.applyCOSFIRE(subject, self.tuples)
		for i in range(self.precision):
			self.rotateTuples()
			for factor in [0.2, 0.5, 0.8, 1, 1.5, 2, 3]:
				tuples = [(rho*factor, phi, *params) for (rho, phi, *params) in self.tuples]
				result = np.maximum(result, self.applyCOSFIRE(subject, tuples))
		return result
		'''

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

	def computeResponses(self, subject):
		responses = {}
		for tupl in self.tuples:
			rho = tupl[0]
			args = tupl[2:]

			# First apply the chosen filter
			filteredResponse = self.filt(*args).transform(subject)

			# Find required values of sigma
			sigmas = [self.sigma0 + rho*self.alpha]
			if self.alpha != 0:
				sigmas = [self.sigma0 + rho*self.alpha for rho in [rho*deltaRho for rho in self.rhoList for deltaRho in self.deltaRhoList]]

			# Apply all blurs
			for sigma in sigmas:
				if ( not((sigma,)+args in responses) ):
					responses[(sigma, )+args] = cosfire.GaussianFilter(sigma).transform(filteredResponse)
		return responses

	def computeTuples(self, deltaPhi, deltaRho):
		return [(rho*deltaRho, phi+(deltaPhi*2*np.pi/self.precision), *params) for (rho, phi, *params) in self.tuples]

	# Function to compute the weighted geometric mean
	# of a list of responses
	def weightedGeometricMean(self, images):
	    maxWeight = 2*(np.amax([img[1] for img in images])/3)**2
	    totalWeight = 0
	    result = np.ones(images[0][0].shape)
	    for img in images:
	        weight = np.exp(-(img[1]**2)/maxWeight)
	        totalWeight += weight
	        result = np.multiply(result, img[0]**weight)
	    return result**(1/totalWeight)
