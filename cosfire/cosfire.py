from sklearn.base import BaseEstimator, TransformerMixin
import cosfire as c
import math as m
import numpy as np
import time
#from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool
import multiprocessing as mp

print("PARALLEL X")

class COSFIRE(BaseEstimator, TransformerMixin):

	def __init__(self, strategy):
		self.strategy = strategy

	def fit(self, *pargs, **kwargs):
		self.strategy.fit(*pargs, **kwargs)
		return self;

	def transform(self, subject):
		return self.strategy.transform(subject)

	def get_params(self, deep=True):
		return self.strategy.get_params(deep)

	def set_params(self, **params):
		return self.strategy.set_params(**params)


class CircleStrategy(BaseEstimator, TransformerMixin):

	def __init__(self, filt, filterArgs, rhoList, prototype, center, sigma0=0, alpha=0, rotationInvariance=[0], scaleInvariance=[1], T1=0, T2=0, numthreads=1):
		self.filterArgs = self.convertFilterArgs(filterArgs) if type(filterArgs) is dict else filterArgs
		self.filt = filt
		self.T1 = T1
		self.T2 = T2
		self.rhoList = rhoList
		self.prototype = prototype
		self.center = center
		self.sigma0 = sigma0/6
		self.alpha = alpha/6
		self.rotationInvariance = rotationInvariance
		self.scaleInvariance = scaleInvariance
		self.timings = []
		self.numthreads = numthreads

	def fit(self):
		self.protoStack = c.ImageStack().push(self.prototype).applyFilter(self.filt, self.filterArgs)
		self.protoStack.treshold = self.T2
		self.tuples = self.findTuples()

	def transform(self, subject):
		t0 = time.time()                                         # Time point
		
		(h,w) = subject.shape
		m = self.maxRho()
		self.pool = Pool(2)
		task1 = self.pool.apply_async(apply, (subject[:int(h/2)+m,:], self.tuples, self.filt, self.sigma0, self.alpha, self.rotationInvariance, self.scaleInvariance) )
		task2 = self.pool.apply_async(apply, (subject[int(h/2)-m:,:], self.tuples, self.filt, self.sigma0, self.alpha, self.rotationInvariance, self.scaleInvariance) )
		result1 = task1.get()
		result2 = task2.get()
		result = np.concatenate( (result1[:int(h/2),:], result2[m:,:]), axis=0 )
		self.pool.close()

		'''
		(h,w) = subject.shape
		m = self.maxRho()
		self.pool = Pool(2)
		task1 = self.pool.apply_async(apply, (subject[:1*int(h/4)+m,:], self.tuples, self.filt, self.sigma0, self.alpha, self.rotationInvariance, self.scaleInvariance) )
		task2 = self.pool.apply_async(apply, (subject[1*int(h/4)-m:2*int(h/4)+m,:], self.tuples, self.filt, self.sigma0, self.alpha, self.rotationInvariance, self.scaleInvariance) )
		task3 = self.pool.apply_async(apply, (subject[2*int(h/4)-m:3*int(h/4)+m,:], self.tuples, self.filt, self.sigma0, self.alpha, self.rotationInvariance, self.scaleInvariance) )
		task4 = self.pool.apply_async(apply, (subject[4*int(h/4)-m:,:], self.tuples, self.filt, self.sigma0, self.alpha, self.rotationInvariance, self.scaleInvariance) )
		result1 = task1.get()
		result2 = task2.get()
		result3 = task3.get()
		result4 = task4.get()
		result = np.concatenate( (result1[:int(h/4),:], result2[m:int(h/4)+m,:], result3[m:int(h/4)+m,:], result4[m:,:]), axis=0 )
		self.pool.close()
		'''

		#result = apply(subject, self.tuples, self.filt, self.sigma0, self.alpha, self.rotationInvariance, self.scaleInvariance)
		self.timings.append( ("\tTransform routines", time.time()-t0) )

		return result

	def divideAndCompute(self, subject, level):
		(h,w) = subject.shape
		m = self.maxRho()
		orientation = h >= w
		if (orientation):
			chunk1 = subject[:int(h/2)+m,:]
			chunk2 = subject[int(h/2)-m:,:]
		else:
			chunk1 = subject[:,:int(w/2)+m]
			chunk2 = subject[:,int(w/2)-m:]
		if (level > 0):
			t0 = time.time()
			if (orientation):
				result = np.concatenate( (self.divideAndCompute(chunk1, level-1)[:int(h/2),:], self.divideAndCompute(chunk2, level-1)[m:,:]), axis=0 )
			else:
				result = np.concatenate( (self.divideAndCompute(chunk1, level-1)[:,:int(w/2)], self.divideAndCompute(chunk2, level-1)[:,m:]), axis=1 )
			self.timings.append( ("\t\tLevel {} recursion of image partitioning".format(level), time.time()-t0) )
			return result
		else:
			t0 = time.time()
			task1 = self.pool.apply_async(apply, (chunk1, self.tuples, self.filt, self.sigma0, self.alpha, self.rotationInvariance, self.scaleInvariance) )
			task2 = self.pool.apply_async(apply, (chunk2, self.tuples, self.filt, self.sigma0, self.alpha, self.rotationInvariance, self.scaleInvariance) )
			result1 = task1.get()
			result2 = task2.get()
			if (orientation):
				result = np.concatenate( (result1[:int(h/2),:], result2[m:,:]), axis=0 )
			else:
				result = np.concatenate( (result1[:,:int(w/2)], result2[:,m:]), axis=1 )
			self.timings.append( ("\t\tComputing 2 chunks of the partitioned image", time.time()-t0) )
			return result

	def findTuples(self):
		# Init some variables
		(cx, cy) = self.center
		tuples = []

		t0 = time.time()                                     # Time point
		# Go over every rho (radius of circles)
		for rho in self.rhoList:
			t1 = time.time()                                 # Time point
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
					phi = (np.arctan2(dy, dx))%(2*np.pi)
					tuples.append( (rho,phi)+vals[i][1] )

			# Store timing
			self.timings.append( ("\tFinding tuples for rho={}".format(rho), time.time()-t1) )

		# Store timing
		self.timings.append( ("Finding all {} tuples for {} different values of rho".format(len(tuples), len(self.rhoList)), time.time()-t0) )

		return tuples

	def maxRho(self):
		return int(np.amax(np.array(self.tuples)[:,0]))+2

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

	def convertFilterArgs(self, dict):
		if len(dict) == 2:
			return (dict['sigma'], dict['onoff'])
		if len(dict) == 5:
			return (dict['sigma'], dict['theta'], dict['lambd'], dict['gamma'], dict['psi'])
		return dict.items()

def computeResponses(subject, tuples, filt, sigma0, alpha, scaleInvariance):

	uniqueArgs = c.unique([ tuple(args) for (rho,phi,*args) in tuples])
	filteredResponses = {}
	for args in uniqueArgs:
		# First apply the chosen filter
		filteredResponse = filt(*args).transform(subject)
		# ReLU
		filteredResponse = np.where(filteredResponse < 0, 0, filteredResponse)
		# Save response
		filteredResponses[args] = filteredResponse

	responses = {}
	for tupl in tuples:
		rho = tupl[0]
		args = tupl[2:]

		if alpha != 0:
			for upsilon in scaleInvariance:
				localRho = rho * upsilon
				localSigma = sigma0 + localRho*alpha
				blurredResponse = c.GaussianFilter(localSigma, sz=int(round(localSigma*6))+(1-int(round(localSigma*6))%2)).transform(filteredResponses[args])
				responses[(localRho,)+args] = blurredResponse
		else:
			blurredResponse = c.GaussianFilter(sigma0).transform(filteredResponses[args])
			for upsilon in scaleInvariance:
				localRho = rho * upsilon
				responses[(localRho,)+args] = blurredResponse

	return responses

def shiftCombine( variation, tuples, shape, responses ):
	psi = variation[0]
	upsilon = variation[1]

	# Adjust base tuples
	curTuples = [(rho*upsilon, phi+psi, *params) for (rho, phi, *params) in tuples]

	# Collect shifted filter responses
	result = np.ones(shape)
	for tupl in curTuples:
		t1 = time.time()                                 # Time point
		rho = tupl[0]
		phi = tupl[1]
		args = tupl[2:]
		dx = int(round(rho*np.cos(phi)))
		dy = int(round(-rho*np.sin(phi)))

		# Apply shift
		result = result * c.shiftImage(responses[(rho,)+args], -dx, -dy).clip(min=0)

	# Combine shifted filter responses
	# curResult = self.weightedGeometricMean(curResponses)
	result = result**(1/len(curTuples))

	return result

def apply(subject, tuples, filt, sigma0, alpha, rotationInvariance, scaleInvariance):

	# Precompute all blurred filter responses
	responses = computeResponses(subject, tuples, filt, sigma0, alpha, scaleInvariance)

	variations = []
	for psi in rotationInvariance:
		for upsilon in scaleInvariance:
			variations.append( (psi, upsilon) )

	# Store the maximum of all the orientations
	result = np.amax([shiftCombine( variation, tuples, subject.shape, responses ) for variation in variations], axis=0)

	return result
