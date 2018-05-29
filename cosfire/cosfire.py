from sklearn.base import BaseEstimator, TransformerMixin
import cosfire as c
import math as m
import numpy as np
import scipy.stats.mstats as mstats
import time
from PIL import Image

class COSFIRE(BaseEstimator, TransformerMixin):

	def __init__(self, strategy, *pargs, **kwargs):
		self.strategy = strategy(*pargs, **kwargs)

	def fit(self, prototype, *pargs, **kwargs):
		self.strategy.fit(prototype, *pargs, **kwargs)
		return self;

	def transform(self, prototype, *pargs, **kwargs):
		return self.strategy.transform(prototype, *pargs, **kwargs)



class CircleStrategy(BaseEstimator, TransformerMixin):

	def __init__(self, filt, filterArgs, rhoList, T1=0.2, sigma0=0, alpha=0, rotationInvariance=[0], scaleInvariance=[1]):
		self.filterArgs = filterArgs
		self.filt = filt
		self.T1 = T1
		self.rhoList = rhoList
		self.sigma0 = sigma0/6
		self.alpha = alpha/6
		self.rotationInvariance = rotationInvariance
		self.scaleInvariance = scaleInvariance

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
		for psi in self.rotationInvariance:  # rotation invariance
			for upsilon in self.scaleInvariance:    # scale invariance
				# Adjust base tuples
				curTuples = [(rho*upsilon, phi+psi, *params) for (rho, phi, *params) in self.tuples]

				# Collect shifted filter responses
				curResponses = []
				for tupl in curTuples:
					rho = tupl[0]
					phi = tupl[1]
					args = tupl[2:]
					dx = int(round(rho*np.cos(phi)))
					dy = int(round(-rho*np.sin(phi)))

					# Apply shift
					response = c.shiftImage(responses[(rho,)+args], -dx, -dy).clip(min=0)

					# Add to set of responses
					curResponses.append( response )

					# Save responses for later reference
					img = Image.fromarray(c.rescaleImage(response, 0, 255).astype(np.uint8))
					img.save("responses/{}_{}_{}.tif".format(rho, c.approx(phi), args[0]))

				# Combine shifted filter responses
				#curResult = self.weightedGeometricMean(curResponses)
				curResult = mstats.gmean(curResponses)

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
		#  - every possible rho (rho*upsilon)
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
			filteredResponse = self.filt(*args).transform(subject).clip(0)

			if self.alpha != 0:
				for upsilon in self.scaleInvariance:
					localRho = rho * upsilon
					blurredResponse = c.GaussianFilter(self.sigma0 + localRho*self.alpha).transform(filteredResponse)
					responses[(localRho,)+args] = blurredResponse
			else:
				blurredResponse = c.GaussianFilter(self.sigma0).transform(filteredResponse)
				for upsilon in self.scaleInvariance:
					localRho = rho * upsilon
					responses[(localRho,)+args] = blurredResponse

		return responses

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
