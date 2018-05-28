from sklearn.base import BaseEstimator, TransformerMixin
import cosfire as c
import math as m
import numpy as np
import scipy.stats.mstats as mstats
import time

class COSFIRE(BaseEstimator, TransformerMixin):

	def __init__(self, strategy, *pargs, **kwargs):
		self.strategy = strategy(*pargs, **kwargs)

	def fit(self, prototype, *pargs, **kwargs):
		self.strategy.fit(prototype, *pargs, **kwargs)
		return self;

	def transform(self, prototype, *pargs, **kwargs):
		return self.strategy.transform(prototype, *pargs, **kwargs)



class CircleStrategy(BaseEstimator, TransformerMixin):

	def __init__(self, filt, filterArgs, rhoList, T1=0.2, sigma0=0, alpha=0, rotationInvariance=12):
		self.filterArgs = filterArgs
		self.filt = filt
		self.T1 = T1
		self.rhoList = rhoList
		self.sigma0 = sigma0
		self.alpha = alpha
		self.rotationInvariance = rotationInvariance
		self.deltaRhoList = [1]

	def fit(self, prototype, center):
		self.prototype = prototype
		self.center = center
		self.protoStack = c.ImageStack().push(prototype).applyFilter(self.filt, self.filterArgs)
		self.protoStack.T1 = self.T1
		self.tuples = self.findTuples()

	def transform(self, subject):

		# Precompute all blurred filter responses
		responses = self.computeResponses(subject)

		# Shifting and putting them together
		def approxTupl(tupl):
			return (tupl[0],c.approx(tupl[1]))+tupl[2:]

		result = np.zeros(subject.shape)
		for deltaPhi in range(self.rotationInvariance):   # rotation invariance
			for deltaRho in self.deltaRhoList:   # scale invariance
				# Adjust base tuples
				curTuples = self.computeTuples(deltaPhi, deltaRho)

				print("")
				for tupl in curTuples:
					print(tupl)

				# Collect shifted filter responses
				curResponses = []
				for tupl in curTuples:
					rho = tupl[0]
					phi = tupl[1]
					args = tupl[2:]
					dx = int(round(rho*np.cos(phi)))
					dy = int(round(rho*np.sin(phi)))
					response = c.shiftImage(responses[(rho,)+args], -dx, -dy).clip(min=0)
					curResponses.append( (response, rho) )

				# Combine shifted filter responses
				#curResult = self.weightedGeometricMean(curResponses)
				curResult = self.weightedGeometricMean(curResponses)

				# Include it in the final result
				result = np.maximum(result, curResult)

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
				# Compute points on the circle of radius rho with center point (cx,cy)
				coords = [ ( cx+int(round(rho*np.cos(phi))) , cy+int(round(rho*np.sin(phi))) )
							for phi in
								[i/360*2*np.pi for i in range(360)]
						 ]
				# Make list unique
				coords = [coords[i] for i in sorted(set([coords.index(c) for c in coords]))]

				# Retrieve values on the circle points in the given filtered prototype
				vals = [self.protoStack.valueAtPoint(*coord)+coord for coord in coords]

				# Find peaks in circle
				maxima = c.circularPeaks([x[0] for x in vals])
				for i in maxima:
					dx = vals[i][2] - cx
					dy = vals[i][3] - cy
					phi = (m.atan2(dy, dx))%(2*m.pi)
					tuples.append( (rho,phi)+vals[i][1] )
				#tuples.extend([ (rho, i*np.pi/360*2)+vals[i][1] for i in maxima])
		return tuples

	def computeResponses(self, subject):
		# We require a response for:
		#  - every possible rho (rho*deltaRho)
		#  - every possible phi (all precision angles)

		# Response steps (all but shifting is interchangable in sequence):
		#  - apply the filter
		#  - apply blurring
		#  - shift the response

		responses = {}
		for tupl in self.tuples:
			rho = tupl[0]
			args = tupl[2:]

			# First apply the chosen filter
			filteredResponse = self.filt(*args).transform(subject)

			if self.alpha != 0:
				for deltaRho in self.deltaRhoList:
					localRho = rho * deltaRho
					blurredResponse = c.GaussianFilter(self.sigma0 + localRho*self.alpha).transform(filteredResponse)
					responses[(localRho,)+args] = blurredResponse
			else:
				blurredResponse = c.GaussianFilter(self.sigma0).transform(filteredResponse)
				for deltaRho in self.deltaRhoList:
					localRho = rho * deltaRho
					responses[(localRho,)+args] = blurredResponse

		return responses

	def computeTuples(self, deltaPhi, deltaRho):
		return [(rho*deltaRho, (phi+(deltaPhi/self.rotationInvariance*np.pi))%(2*np.pi), *params) for (rho, phi, *params) in self.tuples]

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
