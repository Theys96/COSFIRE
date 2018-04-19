from sklearn.base import BaseEstimator, TransformerMixin
import cosfire
import math as m

class COSFIRE(BaseEstimator, TransformerMixin):

	def __init__(self, strategy, *pargs, **kwargs):
		self.strategy = strategy(*pargs, **kwargs)

	def fit(self, prototype, *pargs, **kwargs):
		self.strategy.fit(prototype, *pargs, **kwargs)
		return self;

	def transform(self, prototype, *pargs, **kwargs):
		return self.strategy.transform(prototype, *pargs, **kwargs)



class CircleStrategy(BaseEstimator, TransformerMixin):

	def __init__(self, filt, filterArgs, rhoList, T1=0.2):
		self.sigma = filterArgs[0]
		self.onoff = filterArgs[1]
		self.filter = filt(self.sigma, self.onoff)
		self.T1 = T1
		self.rhoList = rhoList

	def fit(self, prototype, center):
		self.filteredProto = self.filter_suppress(prototype)
		self.tuples = self.findTuples(self.filteredProto, center)
		return self.tuples

	def transform(self, prototype):
		raise NotImplementedError("How does CircleStrategy transform an input?")

	def filter_suppress(self, image):
		return cosfire.SuppressFunction(self.T1).transform(
					self.filter.transform(image)
				)

	def findTuples(self, image, center, precision=16):
		# Init some variables
		(cx, cy) = center
		peakFunction = cosfire.CircularPeaksFunction()
		tuples = []

		# Go over every rho (radius of circles)
		for rho in self.rhoList:
			if rho == 0 and image[cy,cx] > 0:   # Circle with no radius, so just the center point
				tuples.append((self.sigma, rho, 0))
			elif rho > 0:
				# Compute points (amount=precision) on the circle of radius rho with center point (cx,cy)
				coords = [ ( cy+int(round(rho*m.sin(phi))) , cx+int(round(rho*m.cos(phi))) )
							for phi in
								[i*m.pi/precision*2 for i in range(0,precision)]
						 ]
				# Retrieve values on the circle points in the given filtered prototype
				vals = [image[coord] for coord in coords]

				# Find peaks in circle
				maxima = peakFunction.transform(vals)
				tuples.extend([ (self.sigma, rho, i*m.pi/precision*2) for i in maxima])
		return tuples

