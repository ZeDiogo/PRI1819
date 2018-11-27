from data import Data
from textInfo import TextInfo
import spacy
from spacy import displacy
from collections import Counter
# python3 -m spacy download en_core_web_sm #40Mb
# import en_core_web_sm

class Statistics:

	def __init__(self, filename):
		self.data = Data(filename)
		self.texts = self.data.getTexts()
		# self.nlp = spacy.load('en_core_web_sm')

	def getNamedEntities(self, text):
		doc = self.nlp(text)
		idf = {}
		for x in doc.ents:
			if x.label_ in idf:
				idf[x.label_].append(x.text)
			else:
				idf[x.label_] = [x.text]
		return idf
		# return (dict([(x.label_, x.text) for x in doc.ents]))
		# return ([idf.update({x.label_, x.text}) for x in doc.ents])

	def printNamedEntities(self):
		for text in self.texts:
			textInfo = TextInfo(text)
			print(textInfo.getMentionedEntities())
			break

		# doc = self.nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
		# print([(x.text, x.label_) for x in doc.ents])
	    # for text in self.texts:
	    	
	    # 	# print(text)
	    # 	print([(x.text, x.label_) for x in doc.ents])
	    # 	break
		
	def printDict(self, dict):
		for k, v in dict.items():
			print(k, ': ', v)