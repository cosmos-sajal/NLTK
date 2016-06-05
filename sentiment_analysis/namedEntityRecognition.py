import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

custom_sent_tokeniser = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokeniser.tokenize(sample_text)

def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			namedEntity = nltk.ne_chunk(tagged, binary=False)
			namedEntity.draw()
	except Exception as e:
		print str(e)

process_content()