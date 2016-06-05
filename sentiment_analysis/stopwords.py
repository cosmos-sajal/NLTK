from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "This is a sample sentence, showing off the stop words filtration."

stopwords = set(stopwords.words('english'))
word_tokens = word_tokenize(example_sentence)
filtered_sentence = [w for w in word_tokens if w not in stopwords]

print filtered_sentence