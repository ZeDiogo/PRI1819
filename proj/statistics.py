from data import Data
from textInfo import TextInfo
import spacy
from spacy import displacy
from collections import Counter
import time
from pandas import DataFrame
from bar import Bar
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
		print('Building most mentioned entities... (~6min)')
		bar = Bar(self.data.getLength(), timer=True)
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
			# print('Party:', party)
			# print('Party:', party, 'completed in', time.time() - start, 'seconds')
			bar.update()


	def getMentionedEntities(self, party):
		freqs = [(x[1], x[0]) for x in self.mentionsDictionary[party].items()]
		return list(reversed(sorted(freqs, key=lambda x: (isinstance(x, str), x))))

	def getMostMentionedEntities(self, party, minMentions=0, top=100):
		lst = self.getMentionedEntities(party)
		res = [(x[0], x[1]) for x in lst if x[0] > minMentions]
		return res[:top]

	#What are the most mentioned entities for each party?
	def showMostMentionedEntitiesEachParty(self, top=100, minMentions=0):
		for party in self.data.getUniqueParties():
			print('Party:', party, '\nTop', top, ':', self.getMostMentionedEntities(party, top=top, minMentions=minMentions))

	def buildMostMentionedEntitiesGlobally(self):
		globalEntities = {}
		for party in self.data.getUniqueParties():
			for entity, freq in self.mentionsDictionary[party].items():
				if entity in globalEntities:
					globalEntities[entity] += freq
				else:
					globalEntities[entity] = freq
		return globalEntities

	#What are the most mentioned entities globally?
	def showMostMentionedEntitiesGlobally(self, top=100, minMentions=0):
		print('Most mentioned entities globally:')
		globalEntities = self.buildMostMentionedEntitiesGlobally()
		for i, pair in enumerate(globalEntities.items()):
			entity = pair[0]
			freq = pair[1]
			if i >= top:
				break
			if freq < minMentions:
				break
			print(entity, '->', freq)
	
	def printDict(self, dict):
		for k, v in dict.items():
			print(k, ': ', v)