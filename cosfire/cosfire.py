from sklearn.base import BaseEstimator, TransformerMixin
import cosfire

class COSFIRE(BaseEstimator, TransformerMixin):

	def __init__(self, strategy, *pargs, **kwargs):
		self.strategy = strategy(*pargs, **kwargs)

	def fit(self, prototype, *pargs, **kwargs):
		return self.strategy.fit(prototype)

	def transform(self, prototype, *pargs, **kwargs):
		return self.strategy.transform(prototype)



class CircleStrategy(BaseEstimator, TransformerMixin):

	def __init__(self, filt, filterArgs, T1=0.2):
		self.filter = filt(filterArgs[0], filterArgs[1])
		self.T1 = T1

	def fit(self, prototype):
		self.filteredProto = self.filter_suppress(prototype)
		return self.filteredProto

	def transform(self, prototype):
		raise NotImplementedError("How does CircleStrategy transform a prototype?")

	def filter_suppress(self, image):
		return cosfire.SuppressFunction(self.T1).transform(
					self.filter.transform(image)
				)