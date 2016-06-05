from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
example_sentence = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
word_tokens = word_tokenize(example_sentence)
stemmed_words = [ps.stem(w) for w in word_tokens]
print stemmed_words