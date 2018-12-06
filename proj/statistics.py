from data import Data
from textInfo import TextInfo
import spacy
from spacy import displacy
from collections import Counter
import time
from pandas import DataFrame
from bar import Bar
import json
import nltk
import pandas as pd
import jellyfish as jf
# from nltk.metrics import jaccard_distance
# from nltk.metrics import jaro_similarity
# python3 -m spacy download en_core_web_sm #40Mb
# import en_core_web_sm

class Statistics:

	def __init__(self, filename):
		self.data = Data(filename)
		self.texts = self.data.getTexts()
		self.partyDictionary = {}
		self.mentionsDictionary = {}
		self.globalMentionsDictionary = {}
		self.nlp = spacy.load('en_core_web_sm')
		self.buildMostMentionedEntities(useSaved=True)
		self.unique = {}
		self.mentionKeys = {}
		self.initMentionsKeys()
		self.descriptions = {}
		self.initDescriptions()

	# def getNamedEntities(self, text):
	# 	doc = self.nlp(text)
	# 	idf = {}
	# 	for x in doc.ents:
	# 		if x.label_ in idf:
	# 			idf[x.label_].append(x.text)
	# 		else:
	# 			idf[x.label_] = [x.text]
	# 	return id

	# def printNamedEntities(self):
	# 	for text in self.texts:
	# 		textInfo = TextInfo(text)
	# 		print(textInfo.getMostMentionedEntities(mentions=30))
	# 		break

	# def buildPartyDictionary(self):
	# 	for party, text in self.data.getPartiesTexts():
	# 		if party not in self.partyDictionary:
	# 			self.partyDictionary[party] = text
	# 		else:
	# 			self.partyDictionary[party] += ' ' + text

	#In order to improve execution time, the struture generated is stored in a json file
	def buildMostMentionedEntities(self, useSaved=True):
		print('BUILD:', useSaved)
		if useSaved:
			print('Retriving most mentioned entities...')
			start = time.time()
			try:
				with open('mostMentionedEntities.json', 'r') as fp:
					self.mentionsDictionary = json.load(fp)
				print('Finished in', time.time() - start, 'seconds')
			except FileNotFoundError:
				print('Most mentioned entities has not been built yet...')
				useSaved = False
		if not useSaved:
			print('Building most mentioned entities... (~6min)')
			bar = Bar(self.data.getLength(), timer=True)
			for party, text in self.data.getPartiesTexts():
				doc = self.nlp(text)
				for x in doc.ents:
					if party not in self.mentionsDictionary:
						self.mentionsDictionary[party] = {x.label_:[1, [x.text]]}
					elif x.label_ not in self.mentionsDictionary[party]:
						self.mentionsDictionary[party][x.label_] = [1, [x.text]]
					else:
						self.mentionsDictionary[party][x.label_][0] += 1
						self.mentionsDictionary[party][x.label_][1].append(x.text)
				bar.update()
			with open('mostMentionedEntities.json', 'w') as fp:
				json.dump(self.mentionsDictionary, fp)

	def getMentionedEntities(self, party):
		freqs = [(x[1][0], x[0]) for x in self.mentionsDictionary[party].items()]
		return list(reversed(sorted(freqs, key=lambda x: (isinstance(x, str), x))))

	def getMostMentionedEntities(self, party, minMentions=0, top=100):
		lst = self.getMentionedEntities(party)
		res = [(x[0], x[1]) for x in lst if x[0] > minMentions]
		return res[:top]

	#What are the most mentioned entities for each party?
	def showMostMentionedEntitiesEachParty(self, top=5, minMentions=0):
		print()
		print('Parties with top', top, 'named entities:')
		print()
		for party in self.data.getUniqueParties():
			print(party)
			for freq, namedEntity in self.getMostMentionedEntities(party, top=top, minMentions=minMentions):
				print('\t', freq, ':', self.descriptions[namedEntity])

	def buildMostMentionedEntitiesGlobally(self):
		globalEntities = {}
		for party in self.data.getUniqueParties():
			for entity, pair in self.mentionsDictionary[party].items():
				freq = pair[0]
				if entity in globalEntities:
					globalEntities[entity] += freq
				else:
					globalEntities[entity] = freq

		freqs = [(x[1], x[0]) for x in globalEntities.items()]			
		return list(reversed(sorted(freqs, key=lambda x: (isinstance(x, str), x))))

	#What are the most mentioned entities globally?
	def showMostMentionedEntitiesGlobally(self, top=100, minMentions=0):
		print()
		print('Most mentioned entities globally:')
		print()
		print('\tMentions\tNamed Entity')
		print()
		globalEntities = self.buildMostMentionedEntitiesGlobally()
		for i, pair in enumerate(globalEntities):
			freq = pair[0]
			entity = pair[1]
			if i >= top:
				break
			if freq < minMentions:
				break
			print('\t', freq, '\t\t', self.descriptions[entity])

	#Which party is mentioned more times by the other parties?
	def showMostMentionedPartyByOthers(self):
		allParties = self.data.getUniqueParties()
		mentions = self.initDict(allParties)
		print()
		print('Which party mentiones the other ones:')
		print()
		print('\tParty\\Mentioned')
		
		for party in allParties:
			partiesMentioned = self.mentionsDictionary[party]['NORP'][1]
			for partyMentioned in partiesMentioned:
				key = self.matchParty(partyMentioned, allParties)
				if key and key != party: #party x doesnt mention itself
					mentions[party][key] += 1

		for i, p in enumerate(allParties):
			for j, p2 in enumerate(allParties):
				mentions[p]['(' + str(j) + ')' + p2] = mentions[p].pop(p2)
			mentions['(' + str(i) + ')'] = mentions.pop(p)
		dfMentions = pd.DataFrame(mentions)
		print(dfMentions.to_string())

	def initDict(self, parties):
		return {p:{party:0 for party in parties} for p in parties}

	def matchParty(self, mention, allParties):
		# self.unique[mention] = 1
		mention = mention.lower().strip()
		if len(mention) == 0: return False #spaces

		#using manual references
		if mention in self.mentionKeys:
			self.saveMatch('matches.txt', 'Using etiquetas: ' + str(mention) + ' -> ' + str(self.mentionKeys[mention]))
			return self.mentionKeys[mention]
		
		for party in allParties:
			keyParty = party.lower().strip()

			#Using distance for full title
			if jf.levenshtein_distance(mention, keyParty) < 5:
				self.saveMatch('matches.txt', 'Full title jaro distance: ' + str(mention) + ' -> ' + str(party))
				return party

			#Using distance for single word in party title
			for word in keyParty.split():
				# if jf.jaro_distance(mention, word) < 0.2:
				# 	self.saveMatch('Single word jaro distance: ' + str(mention) + ' -> ' + str(party))
				# 	return party
				if jf.levenshtein_distance(mention, word) < 3:
					self.saveMatch('matches.txt', 'Single word levenstein distance: ' + str(mention) + ' -> ' + str(party))
					return party
			# print('Party:', party, '| keyParty:', keyParty, '| levenshtein_dist:', jf.levenshtein_distance(party, keyParty), '| jaro:', jf.jaro_distance(party, keyParty))
			# self.saveMatch('Mention:' + str(mention) + '| keyParty:' + str(keyParty) + '| levenshtein_dist:' + str(jf.levenshtein_distance(mention, keyParty)) + '| jaro:' + str(jf.jaro_distance(mention, keyParty)))
		
		return False

	def saveMatch(self, filename, line):
		with open(filename, 'a') as fp:
			fp.write(line + '\n')

	def initMentionsKeys(self):
		file = pd.read_csv('mentionsTrain.csv', sep=',', header=0)
		for key, party in zip(file['key'].tolist(), file['party'].tolist()):
			self.mentionKeys[key] = party

	def initDescriptions(self):
		file = pd.read_csv('spacyNamedEntities.csv', sep=';', header=0)
		for namedEntity, description in zip(file['namedEntity'].tolist(), file['description'].tolist()):
			self.descriptions[namedEntity] = description



	# def printDict(self, dict):
	# 	for k, v in dict.items():
	# 		print(k, ': ', v)