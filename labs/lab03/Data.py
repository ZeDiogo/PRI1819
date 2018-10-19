from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import stop_words
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC

class Data:
	def __init__(self):
		self.train = fetch_20newsgroups(subset='train')
		self.test = fetch_20newsgroups(subset='test')
		# print "-------__> TRAIN"
		# print self.train.data[:10]
		# print '\n\n--------> TEST'
		# print self.test.data[:10]

	def createVectors(self):
		# vectorizer = TfidfVectorizer(use_idf=False)
		# vectorizer = TfidfVectorizer(use_idf=False, stop_words='english')
		vectorizer = TfidfVectorizer(use_idf=False, stop_words='english', min_df=3, max_df=0.9)
		self.trainvec = vectorizer.fit_transform(self.train.data)
		self.testvec = vectorizer.transform(self.test.data)

	def createClassifier(self):
		# classifier = MultinomialNB()
		# classifier = KNeighborsClassifier(n_neighbors=10)
		# classifier = Perceptron()
		classifier = LinearSVC()

		classifier.fit(self.trainvec, self.train.target)
		print self.trainvec
		print "\n\n\n"
		print self.train.target
		self.classes = classifier.predict(self.testvec)

	def printResults(self):
		print metrics.accuracy_score(self.test.target, self.classes)
		print metrics.classification_report(self.test.target, self.classes)

	def kmeansClustering(self):
		self.cluster = MiniBatchKMeans(20)
		self.cluster.fit(self.trainvec)





# knn = KNeighborsClassifier()
# 		hyperparamenters = {
# 			"n_neighbors": range(1, 10, 2)
# 		}
# 		grid = GridSearchCV(knn, param_grid=hyperparamenters, cv=10)
# 		grid.fit(self.trainvec, self.train.target)
# 		print(grid.best_params_)
# 		print(grid.best_score_)