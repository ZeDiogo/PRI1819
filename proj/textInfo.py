import spacy
from spacy import displacy
import time

class TextInfo:

	def __init__(self, text):
		self.text = text
		self.wordDictionary = {}
		self.entityDictionary = {}
		self.nlp = spacy.load('en_core_web_sm')
		# self.buildWordDictionary()
		self.buildEntityDictionary()

	def getText(self):
		return self.text

	def buildWordDictionary(self):
		start = time.time()
		doc = self.nlp(self.text)
		for x in doc.ents:
			if x.text in self.wordDictionary:
				self.wordDictionary[x.text][0] += 1
			else:
				self.wordDictionary[x.text] = [1, x.label_]
		print('buildWordDictionary took', time.time() - start, 'seconds to build')

	def buildEntityDictionary(self):
		start = time.time()
		doc = self.nlp(self.text)
		for x in doc.ents:
			if x.label_ in self.entityDictionary:
				self.entityDictionary[x.label_][0] += 1
				# if x.text not in self.entityDictionary[x.label_]:
				# 	self.entityDictionary[x.label_].append(x.text)
			else:
				self.entityDictionary[x.label_] = [1]#s, x.text]
		print('buildEntityDictionary took', time.time() - start, 'seconds')

	def getEntityFrequency(self, entity):
		return self.entityDictionary[entity][0]

	def getWordsFromEntity(self, entity):
		return self.entityDictionary[entity][1:]

	def getWordFrequency(self, word):
		return self.wordDictionary[word][0]

	def getMentionedEntities(self):
		freqs = [(x[1][0], x[0]) for x in self.entityDictionary.items()]
		return list(reversed(sorted(freqs, key=lambda x: (isinstance(x, str), x))))

	def getMostMentionedEntities(self, mentions=15):
		lst = self.getMentionedEntities()
		return [(x[0], x[1]) for x in lst if x[0] > mentions]

	def printDict(self, dict):
		for k, v in dict.items():
			print(k, ': ', v)
