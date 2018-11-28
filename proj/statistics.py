from data import Data
from textInfo import TextInfo
import spacy
from spacy import displacy
from collections import Counter
import time
from pandas import DataFrame
# python3 -m spacy download en_core_web_sm #40Mb
# import en_core_web_sm

class Statistics:

	def __init__(self, filename):
		self.data = Data(filename)
		self.texts = self.data.getTexts()
		self.partyDictionary = {}
		self.mentionsDictionary = {}
		self.globalMentionsDictionary = {}
		# self.partyList = []
		self.nlp = spacy.load('en_core_web_sm')
		self.buildMostMentionedEntities()

	def getNamedEntities(self, text):
		doc = self.nlp(text)
		idf = {}
		for x in doc.ents:
			if x.label_ in idf:
				idf[x.label_].append(x.text)
			else:
				idf[x.label_] = [x.text]
		return id

	def printNamedEntities(self):
		for text in self.texts:
			textInfo = TextInfo(text)
			print(textInfo.getMostMentionedEntities(mentions=30))
			break

	def buildPartyDictionary(self):
		for party, text in self.data.getPartiesTexts():
			if party not in self.partyDictionary:
				self.partyDictionary[party] = text
			else:
				self.partyDictionary[party] += ' ' + text

	def buildMostMentionedEntities(self):
		startTotal = time.time()
		print('Start building most mentioned entities')
		# self.buildPartyDictionary()
		for party, text in self.data.getPartiesTexts():
			doc = self.nlp(text)
			# start = time.time()
			for x in doc.ents:
				if party not in self.mentionsDictionary:
					self.mentionsDictionary[party] = {x.label_:1}
				elif x.label_ not in self.mentionsDictionary[party]:
					self.mentionsDictionary[party][x.label_] = 1
				else:
					self.mentionsDictionary[party][x.label_] += 1
			print('Party:', party)
			# print('Party:', party, 'completed in', time.time() - start, 'seconds')
		print('Builded Most Mentioned Entities in', time.time() - startTotal, 'seconds')

	def getMentionedEntities(self, party):
		freqs = [(x[1], x[0]) for x in self.mentionsDictionary[party].items()]
		return list(reversed(sorted(freqs, key=lambda x: (isinstance(x, str), x))))

	def getMostMentionedEntities(self, party, mentions=15):
		lst = self.getMentionedEntities(party)
		return [(x[0], x[1]) for x in lst if x[0] > mentions]

	# def buildPartyDictionary(self):
	# 	start = time.time()
	# 	for party, text in self.data.getPartiesTexts():
	# 		print(party)
	# 		if party not in self.partyDictionary:
	# 			self.partyDictionary[party] = [TextInfo(text)]
	# 		else:
	# 			self.partyDictionary[party].append(TextInfo(text))
	# 	print('buildEntityDictionary took', time.time() - start, 'seconds')
		
	def printDict(self, dict):
		for k, v in dict.items():
			print(k, ': ', v)