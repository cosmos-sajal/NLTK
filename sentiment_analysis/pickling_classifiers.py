import nltk
import pickle
import random
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI

short_pos = open('../datasets/short_reviews/positive.txt', 'r').read()
short_neg = open('../datasets/short_reviews/negative.txt', 'r').read()

all_words = []
documents = []
allowed_word_types = ['J']

# adding positive words to all Words list
for p in short_pos.split('\n'):
	p = p.decode('utf-8')
	documents.append((p, 'pos'))
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	all_words += [w[0].lower() for w in pos if w[1][0] in allowed_word_types]

# adding negative words to all Words list
for p in short_neg.split('\n'):
	p = p.decode('utf-8')
	documents.append((p, 'neg'))
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	all_words += [w[0].lower() for w in pos if w[1][0] in allowed_word_types]

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

def pickling(file, document_name):
	save_documemts = open('../pickled_algos/' + document_name + '.pickle', 'wb')
	pickle.dump(file, save_documemts)
	save_documemts.close()

def find_features(sentence):
	words = word_tokenize(sentence)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)

# segregating training and testing set
testing_set = featuresets[4300:]
training_set = featuresets[:4300]

# classification using classifiers
def classify():
	# Original Naive Bayes
	classifier = nltk.NaiveBayesClassifier.train(training_set)
	print("Original Naive Bayes Classifer Accuracy:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
	pickling(classifier, 'original_naive_bayes_classifer')

	# Multinomial Naive Bayes
	MNB_classifier = SklearnClassifier(MultinomialNB())
	MNB_classifier.train(training_set)
	print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))  * 100)
	pickling(MNB_classifier, 'MNB_classifier')

	# Bernouli Naive Bayes
	Bernoulli_classifier = SklearnClassifier(BernoulliNB())
	Bernoulli_classifier.train(training_set)
	print("Bernoulli_classifier accuracy percent:", (nltk.classify.accuracy(Bernoulli_classifier, testing_set))  * 100)
	pickling(Bernoulli_classifier, 'Bernoulli_classifier')

	# Logistic Regression Classifier
	LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
	LogisticRegression_classifier.train(training_set)
	print("LogisticRegression_classifier accuracy percent:", \
		(nltk.classify.accuracy(LogisticRegression_classifier, testing_set))  * 100)
	pickling(LogisticRegression_classifier, 'LogisticRegression_classifier')

	# Linear SVC Classifer
	LinearSVC_classifier = SklearnClassifier(LinearSVC())
	LinearSVC_classifier.train(training_set)
	print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))  * 100)
	pickling(LinearSVC_classifier, 'LinearSVC_classifier')

	# NuSVC Classigier
	NuSVC_classifier = SklearnClassifier(NuSVC())
	NuSVC_classifier.train(training_set)
	print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))  * 100)
	pickling(NuSVC_classifier, 'NuSVC_classifier')

	# SGDC Classifier
	SGDC_classifier = SklearnClassifier(SGDClassifier())
	SGDC_classifier.train(training_set)
	print("SGDC_classifier accuracy percent:", (nltk.classify.accuracy(SGDC_classifier, testing_set))  * 100)
	pickling(SGDC_classifier, 'SGDC_classifier')

classify()

# pickling
pickling(documents, 'documents')
pickling(word_features, 'word_features5k')

















