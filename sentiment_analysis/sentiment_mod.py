import nltk
import pickle
import random
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode

featuresets = []
classifier = []
word_features = []
LinearSVC_classifier = []
MNB_classifier = []
Bernoulli_classifier = []
LogisticRegression_classifier = []
NuSVC_classifier = []
SGDC_classifier = []

# class VoteClassifier(ClassifierI):
#     def __init__(self, *classifiers):
#         self._classifiers = classifiers

#     def classify(self, features):
#         votes = []
#         for c in self._classifiers:
#             v = c.classify(features)
#             votes.append(v)
#         return mode(votes)

#     def confidence(self, features):
#         votes = []
#         for c in self._classifiers:
#             v = c.classify(features)
#             votes.append(v)

#         choice_votes = votes.count(mode(votes))
#         conf = choice_votes / len(votes)
#         return conf

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers
		self._votes = []

	def __calculate_votes__(self, features):
		self._votes = [c.classify(features) for c in self._classifiers]
		return self._votes

	def classify(self, features):
		self._votes = [c.classify(features) for c in self._classifiers]
		return mode(self._votes)

	def confidence(self, features):
		choice_votes = self._votes.count(mode(self._votes))
		conf = choice_votes / len(self._votes)
		return conf

def loadAllFiles():
	global featuresets, classifier, word_features, LinearSVC_classifier, MNB_classifier, Bernoulli_classifier, \
		LogisticRegression_classifier, NuSVC_classifier, SGDC_classifier

	# loading documents
	document_f = open('../pickled_algos/documents.pickle', 'rb')
	documents = pickle.load(document_f)
	document_f.close()

	# loading word_features
	word_features_f = open('../pickled_algos/word_features5k.pickle', 'rb')
	word_features = pickle.load(word_features_f)
	word_features_f.close()

	# loading featuresets
	featuresets_f = open('../pickled_algos/featuresets.pickle', 'rb')
	featuresets = pickle.load(featuresets_f)
	featuresets_f.close()

	# loading classifiers
	# Original NB
	classifier_f = open('../pickled_algos/original_naive_bayes_classifer.pickle', 'rb')
	classifier = pickle.load(classifier_f)
	classifier_f.close()

	# MNB Classifier
	MNB_classifier_f = open('../pickled_algos/MNB_classifier.pickle', 'rb')
	MNB_classifier = pickle.load(MNB_classifier_f)
	MNB_classifier_f.close()

	# Bernoulli_classifier
	Bernoulli_classifier_f = open('../pickled_algos/Bernoulli_classifier.pickle', 'rb')
	Bernoulli_classifier = pickle.load(Bernoulli_classifier_f)
	Bernoulli_classifier_f.close()

	# LogisticRegression_classifier
	LogisticRegression_classifier_f = open('../pickled_algos/LogisticRegression_classifier.pickle', 'rb')
	LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_f)
	LogisticRegression_classifier_f.close()

	# LinearSVC_classifier
	LinearSVC_classifier_f = open('../pickled_algos/LinearSVC_classifier.pickle', 'rb')
	LinearSVC_classifier = pickle.load(LinearSVC_classifier_f)
	LinearSVC_classifier_f.close()

	# NuSVC_classifier
	NuSVC_classifier_f = open('../pickled_algos/NuSVC_classifier.pickle', 'rb')
	NuSVC_classifier = pickle.load(NuSVC_classifier_f)
	NuSVC_classifier_f.close()

	# SGDC_classifier
	SGDC_classifier_f = open('../pickled_algos/SGDC_classifier.pickle', 'rb')
	SGDC_classifier = pickle.load(SGDC_classifier_f)
	SGDC_classifier_f.close()

def find_features(document):
    global word_features
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# load all pickled data
loadAllFiles()
random.shuffle(featuresets)

# segregating training and testing set
testing_set = featuresets[4300:]
training_set = featuresets[:4300]

voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  Bernoulli_classifier,
                                  LogisticRegression_classifier,
                                  NuSVC_classifier,
                                  SGDC_classifier)
	
def sentiment(text):
	features = find_features(text)
	return voted_classifier.classify(features), voted_classifier.confidence(features)
