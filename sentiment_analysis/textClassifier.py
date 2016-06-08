import nltk
import random
import pickle
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileId)), category)
			for category in movie_reviews.categories()
			for fileId in movie_reviews.fileids(category)]

random.shuffle(documents)
# most common 3000 words in the movie reviews documents
word_features = list(nltk.FreqDist([w.lower() for w in movie_reviews.words()]).keys())[:3000]

def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# dividing the data set into training and testing set
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Naive Bayes Classifier
# classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier_f = open('naivebayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

accuracy = nltk.classify.accuracy(classifier, testing_set) * 100

print accuracy

# save classifier
# save_classifier = open('naivebayes.pickle', 'wb')
# pickle.dump(classifier, save_classifier)
# save_classifier.close()
