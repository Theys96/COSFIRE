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

	def __init__(self, filt, filterArgs, rhoList, T1=0.2, sigma0=0, alpha=0, precision=24):
		self.filterArgs = filterArgs
		self.filt = filt
		self.T1 = T1
		self.rhoList = rhoList
		self.sigma0 = sigma0
		self.alpha = alpha
		self.precision = precision
		self.deltaRhoList = [0.3, 0.5, 0.8, 1, 1.5]

	def fit(self, prototype, center):
		self.prototype = prototype
		self.center = center
		self.protoStack = c.ImageStack().push(prototype).applyFilter(self.filt, self.filterArgs)
		self.protoStack.T1 = self.T1
		self.tuples = self.findTuples()

	def transform(self, subject):

		# Precompute all blurred filter responses
		print("Pre-computing responses")
		responses = self.computeResponses(subject)
		print("Responses pre-computed")

		# Shifting and putting them together
		def approxTupl(tupl):
			return (tupl[0],c.approx(tupl[1]))+tupl[2:]

		result = np.zeros(subject.shape)
		for deltaPhi in range(self.precision):   # rotation invariance
			for deltaRho in self.deltaRhoList:   # scale invariance
				print("Doing Δphi=", deltaPhi, " Δrho=", deltaRho)
				# Adjust base tuples
				curTuples = self.computeTuples(deltaPhi, deltaRho)

				#start = time.time()
				# Collect shifted filter responses
				curResponses = []
				for tupl in curTuples:
					#curResponses.append((responses[approxTupl(tupl)], tupl[0]))
					curResponses.append(responses[approxTupl(tupl)])
				#responseCollect = time.time()

				# Combine shifted filter responses
				#curResult = self.weightedGeometricMean(curResponses)
				curResult = mstats.gmean(curResponses)
				#calcMean = time.time()

				# Include it in the final result
				result = np.maximum(result, curResult)
				#calcMax = time.time()
				'''
				print("done:")
				print("\tTime to collect filter responses: ", round((responseCollect - start)*1000,3), "ms")
				print("\tTime to compute the weighted geometric mean: ", round((calcMean - responseCollect)*1000,3), "ms")
				print("\tTime to add the result to the main result: ", round((calcMax - calcMean)*1000,3), "ms")
				print("\t\tTotal time for this step: ", round((calcMax - start)*1000,3), "ms")
				'''

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
				maxima = c.circularPeaks([x[0] for x in vals])
				tuples.extend([ (rho, i*np.pi/self.precision*2)+vals[i][1] for i in maxima])
		return tuples

	def computeResponses(self, subject):
		# We require a response for:
		#  - every possible rho (rho*deltaRho)
		#  - every possible phi (all precision angles)

		# Response steps (all but shifting is interchangable in sequence):
		#  - apply the filter
		#  - apply blurring
		#  - shift the response

		rhoResponses = {}
		for tupl in self.tuples:
			rho = tupl[0]
			args = tupl[2:]

			# First apply the chosen filter
			filteredResponse = self.filt(*args).transform(subject)

			if self.alpha != 0:
				for deltaRho in self.deltaRhoList:
					localRho = rho * deltaRho
					blurredResponse = c.GaussianFilter(self.sigma0 + localRho*self.alpha).transform(filteredResponse)
					rhoResponses[(localRho,)+args] = blurredResponse
			else:
				blurredResponse = c.GaussianFilter(self.sigma0).transform(filteredResponse)
				for deltaRho in self.deltaRhoList:
					localRho = rho * deltaRho
					rhoResponses[(localRho,)+args] = blurredResponse

		responses = {}
		for tupl in rhoResponses:
			rho = tupl[0]
			args = tupl[1:]
			step = 2*np.pi/self.precision
			for phi in range(self.precision+1):
				phi = c.approx(step*phi)
				dx = int(round(rho*np.cos(phi)))
				dy = int(round(rho*np.sin(phi)))
				responses[(rho,phi)+args] = c.shiftImage(rhoResponses[tupl], -dx, -dy).clip(min=0)


		return responses

	def computeTuples(self, deltaPhi, deltaRho):
		return [(rho*deltaRho, (phi+(deltaPhi*2*np.pi/self.precision))%(2*np.pi), *params) for (rho, phi, *params) in self.tuples]

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
