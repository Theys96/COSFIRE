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

	def __init__(self, filt, filterArgs, rhoList, T1=0.2):
		self.filterArgs = filterArgs
		self.filt = filt
		self.T1 = T1
		self.rhoList = rhoList

	def fit(self, prototype, center):
		self.protoStack = cosfire.ImageStack(prototype, self.filt, self.filterArgs, self.T1)
		self.tuples = self.findTuples(self.protoStack, center)

	def transform(self, prototype):
		raise NotImplementedError("How does CircleStrategy transform an input?")

	def findTuples(self, image, center, precision=16):
		# Init some variables
		(cx, cy) = center
		peakFunction = cosfire.CircularPeaksFunction()
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
				coords = [ ( cx+int(round(rho*m.cos(phi))) , cy+int(round(rho*m.sin(phi))) )
							for phi in
								[i*m.pi/precision*2 for i in range(0,precision)]
						 ]
				# Retrieve values on the circle points in the given filtered prototype
				vals = [self.protoStack.valueAtPoint(*coord) for coord in coords]

				# Find peaks in circle
				maxima = peakFunction.transform([x[0] for x in vals])
				tuples.extend([ (rho, i*m.pi/precision*2)+vals[i][1] for i in maxima])
		return tuples

