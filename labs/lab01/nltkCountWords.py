import nltk
from prettyPrint import prettyPrint
from nltk.tokenize import TreebankWordTokenizer as tbwt


def nltkCountWords(FILE):
	dictionary = {}

	with open(FILE) as file:
		doc = file.read()
		# print tbwt().tokenize(doc)
		for sentence in nltk.sent_tokenize(doc):
			for word in nltk.word_tokenize(sentence):
				#print nltk.pos_tag(word)
				if dictionary.has_key(word):
					dictionary[word] += 1
				else:
					dictionary[word] = 1
	file.close()
	#prettyPrint(dictionary)
	return dictionary


def main():
	nltkCountWords("text.txt")
	

if __name__ == "__main__":
	main()

