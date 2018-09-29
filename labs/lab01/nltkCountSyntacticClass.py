import nltk
from prettyPrint import prettyPrint
from nltk.tokenize import TreebankWordTokenizer as tbwt

FILE = "text.txt"

dictionary = {}

with open(FILE) as file:
	doc = file.read()
	
	for sentence in nltk.sent_tokenize(doc):
		for tag in nltk.pos_tag(nltk.word_tokenize(sentence)):
			if dictionary.has_key(tag[1]):
				dictionary[tag[1]] += 1
			else:
				dictionary[tag[1]] = 1

# print(dictionary)
prettyPrint(dictionary)