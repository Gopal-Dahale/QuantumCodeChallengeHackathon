from sklearn.tree import DecisionTreeClassifier
import numpy as np
from dimod import BinaryQuadraticModel

class WeakClassifiers(object):
	"""
	Weak Classifiers based on DecisionTree
	"""

	def __init__(self, n_estimators=50, max_depth=3):
		self.n_estimators = n_estimators
		self.estimators_ = []
		self.max_depth = max_depth
		self.__construct_wc()

	def __construct_wc(self):

		self.estimators_ = [
			DecisionTreeClassifier(
				max_depth=self.max_depth,
				random_state=np.random.randint(low=1e6, high=1e7),
			)
			for _ in range(self.n_estimators)
		]

	def fit(self, X, y):
		"""
		fit estimators
		:param X:
		:param y:
		:return:
		"""

		self.estimator_weights = np.zeros(self.n_estimators)

		d = np.ones(len(X)) / len(X)
		for i, h in enumerate(self.estimators_):
			h.fit(X, y, sample_weight=d)
			pred = h.predict(X)
			eps = d.dot(pred != y)
			if eps == 0:  # to prevent divided by zero error
				eps = 1e-20
			w = (np.log(1 - eps) - np.log(eps)) / 2
			d = d * np.exp(-w * y * pred)
			d = d / d.sum()
			self.estimator_weights[i] = w

	def predict(self, X):
		"""
		predict label of X
		:param X:
		:return:
		"""

		if not hasattr(self, "estimator_weights"):
			raise Exception("Not Fitted Error!")

		y = np.zeros(len(X))

		for h, w in zip(self.estimators_, self.estimator_weights):
			y += w * h.predict(X)

		y = np.sign(y)

		return y

	def copy(self):

		classifier = WeakClassifiers(
			n_estimators=self.n_estimators, max_depth=self.max_depth
		)
		classifier.estimators_ = deepcopy(self.estimators_)
		if hasattr(self, "estimator_weights"):
			classifier.estimator_weights = np.array(self.estimator_weights)

		return classifier


class QBoostClassifier(WeakClassifiers):
	"""
	Qboost Classifier
	"""

	def __init__(self, n_estimators=50, max_depth=3):
		super(QBoostClassifier, self).__init__(
			n_estimators=n_estimators, max_depth=max_depth
		)

	def fit(self, X, y, sampler, lmd=0.2, print_bqm_info=False, **kwargs):

		n_data = len(X)

		# step 1: fit weak classifiers
		super(QBoostClassifier, self).fit(X, y)

		# step 2: create QUBO
		hij = []
		for h in self.estimators_:
			hij.append(h.predict(X))

		hij = np.array(hij)
		# scale hij to [-1/N, 1/N]
		hij = 1.0 * hij / self.n_estimators

		## Create QUBO
		qii = n_data * 1.0 / (self.n_estimators**2) + lmd - 2 * np.dot(hij, y)
		qij = np.dot(hij, hij.T)
		Q = dict()
		Q.update(dict(((k, k), v) for (k, v) in enumerate(qii)))
		for i in range(self.n_estimators):
			for j in range(i + 1, self.n_estimators):
				Q[(i, j)] = qij[i, j]

		bqm = BinaryQuadraticModel.from_qubo(Q)
		if print_bqm_info:
			print("BQM info:")
			print("Num variables:", bqm.num_variables)
			print("Num interactions:", bqm.num_interactions)

		# step 3: optimize BQM
		res = sampler.sample(bqm, **kwargs)
		samples = np.array(
			[[samp[k] for k in range(self.n_estimators)] for samp in res]
		)

		# take the optimal solution as estimator weights
		self.estimator_weights = samples[0]

	def predict(self, X):
		n_data = len(X)
		pred_all = np.array([h.predict(X) for h in self.estimators_])
		temp1 = np.dot(self.estimator_weights, pred_all)
		T1 = np.sum(temp1, axis=0) / (n_data * self.n_estimators * 1.0)
		y = np.sign(temp1 - T1)  # binary classes are either 1 or -1

		return y
