from data import Data
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
import operator
import numpy
import re
import matplotlib.pyplot as plt

class Statistics:

	def __init__(self, filename):
		self.totalMentions = 0 #aux
		self.data = Data(filename)
		self.texts = self.data.getTexts()
		self.partyDictionary = {}
		self.mentionsDictionary = {}
		self.globalMentionsDictionary = {}
		self.nlp = spacy.load('en_core_web_sm')
		self.buildIndexNamedEntities(useSaved=True)
		self.unique = {}
		self.mentionKeys = {}
		self.initMentionsKeys()
		self.descriptions = {}
		self.initDescriptions()
		self.buildMentionedPartyByOthers()
		self.tradutor = {}
		

	#In order to improve execution time, the struture generated is stored in a json file
	def buildIndexNamedEntities(self, useSaved=True):
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

	def getMentionedEntitiesOrderByNamedEntity(self, party):
		l = self.getMentionedEntities(party)
		#need simular vectors
		allNamedEntities = self.descriptions.keys()
		for namedEntity in allNamedEntities:
			if namedEntity not in self.mentionsDictionary[party].keys():
				l.append((0, namedEntity))
		ordered = sorted(l, key=lambda x: x[1])
		return [x[0] for x in ordered]

	def getMostMentionedEntities(self, party, minMentions=0, top=100):
		lst = self.getMentionedEntities(party)
		res = [(x[0], x[1]) for x in lst if x[0] > minMentions]
		return res[:top]

	#What are the most mentioned entities for each party?
	def showMostMentionedEntitiesEachParty(self, top=3, minMentions=0):
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

		return self.dict2listOrderedByValue(globalEntities)

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
	
	def buildMentionedPartyByOthers(self):
		allParties = self.data.getUniqueParties()
		self.mentions = self.initDict(allParties)
		self.totalPartyMentions = 0
		for party in allParties:
			partiesMentioned = self.mentionsDictionary[party]['NORP'][1]
			for partyMentioned in partiesMentioned:
				key = self.matchParty(partyMentioned, allParties)
				if key and key != party: #party x doesnt mention itself
					self.mentions[party][key] += 1
					self.totalPartyMentions += 1


	#show matrix of mentions
	def showMentionsMatrix(self):
		allParties = self.data.getUniqueParties()
		print()
		print('Which party mentiones the other ones:')
		print()
		print('\tMentioned Party\\Party who mentioned')
		m = {}
		for i, p in enumerate(allParties):
			m['(' + str(i).zfill(2) + ')'] = {}
			for j, p2 in enumerate(allParties):
				m['(' + str(i).zfill(2) + ')']['(' + str(j).zfill(2) + ')' + p2] = self.mentions[p][p2]
			# mentions['(' + str(i) + ')'] = mentions.pop(p)
		dfMentions = pd.DataFrame(m)
		print(dfMentions.to_string())
		dfMentions.plot(kind='bar')
		plt.show()

	#Which party is mentioned more times by the other parties?
	def showMostMentionedParty(self):
		allParties = self.data.getUniqueParties()
		totalMentions = {}
		for whoWasMentioned in allParties:
			totalMentions[whoWasMentioned] = 0
			for whoMentioned in allParties:
				totalMentions[whoWasMentioned] += self.mentions[whoMentioned][whoWasMentioned]
		print()
		print('Most mentioned party:')
		mostMentioned = max(totalMentions.items(), key=operator.itemgetter(1))[0]
		print('\t', mostMentioned, totalMentions[mostMentioned], 'times')
		print('\t\tMentioned by:')
		maxLength = self.data.getMaxLengthParty()
		forOrder = []
		for whoMentioned in allParties:
			forOrder.append((self.mentions[whoMentioned][mostMentioned], whoMentioned))

		for refs, whoMentioned in list(reversed(sorted(forOrder, key=lambda x: (isinstance(x, str), x)))):
			spacing = maxLength-len(whoMentioned)
			print('\t\t\t', whoMentioned, ' '*spacing, refs, 'times')

	def sumAllValues(self, d):
		total = 0
		for k, v in d.items():
			total += v
		return total

	def dict2listOrderedByValue(self, d):
		freqs = [(x[1], x[0]) for x in d.items()]
		return list(reversed(sorted(freqs, key=lambda x: (isinstance(x, str), x))))

	#How many times does any given party mention other parties?
	def showHowManyTimesEachPartyMentionsOthers(self):
		print()
		print('How many times each party mentiones others:')
		print()
		allParties = self.data.getUniqueParties()
		totalMentions = {}
		for whoMentioned in allParties:
			totalMentions[whoMentioned] = self.sumAllValues(self.mentions[whoMentioned])

		maxLength = self.data.getMaxLengthParty()
		for freq, whoMentioned in self.dict2listOrderedByValue(totalMentions):
			spacing = maxLength-len(whoMentioned)
			print('\t', whoMentioned, ' '*spacing, freq, 'times')

	def Edistance(self,v1,v2):
		a = numpy.array(v1)
		b = numpy.array(v2)
		return numpy.sqrt(sum((a-b)**2))

	def buildDistanceBetweenNamedEntities(self):
		allParties = self.data.getUniqueParties()
		vectors = {}
		for party in allParties:
			vectors[party] = self.getMentionedEntitiesOrderByNamedEntity(party)

		distances = {}
		for i, p1 in enumerate(allParties):
			key = '(' + str(i).zfill(2) + ')'
			distances[key] = {}
			self.tradutor[key] = p1
			for j, p2 in enumerate(allParties):
				if i <= j:
					key2 = '(' + str(j).zfill(2) + ')' + p2
					distances[key][key2] = '%.2f' % self.Edistance(vectors[p1], vectors[p2])
					self.tradutor[key2] = p2
		return distances

	#Vector distance between named entities vectores for each party
	def showDistanceBetweenNamedEntities(self):
		distances = self.buildDistanceBetweenNamedEntities()
		df = pd.DataFrame(distances)
		print()
		print('Euclidean Distance between named entities:')
		print()
		print(df.to_string())

	def showMostSimilarParties(self, threshold=500):
		allParties = self.data.getUniqueParties()
		distances = self.buildDistanceBetweenNamedEntities()
		
		simScores = []
		for p1 in distances.keys():
			# simScores[p1] = {}
			for p2 in distances[p1].keys():
				# simScores[p1][p2] = 1/float(distances[p1][p2])
				dist = float(distances[p1][p2])
				print(dist)
				if dist < threshold and dist > 0:
					simScores.append((dist, self.tradutor[p1] + ' <--> ' + self.tradutor[p2]))

		print()
		print('Similiarity Score between named entities:')
		print()
		maxLength = self.data.getMaxLengthParty()*2+6
		spacing = maxLength-len('Parties')
		print('\tParties' + ' '*spacing + 'Distance')
		print()
		for dist, parties in sorted(simScores, key=lambda x: x[0]):
			spacing = maxLength-len(parties)
			print('\t', parties, ' '*spacing, dist)

	def initDict(self, parties):
		return {p:{party:0 for party in parties} for p in parties}

	def matchParty(self, mention, allParties):
		mention = mention.lower().strip()
		if len(mention) == 0: return False #spaces
		self.totalMentions += 1
		#using manual references
		if mention in self.mentionKeys:
			# self.saveMatch('matches.txt', 'Using etiquetas: ' + str(mention) + ' -> ' + str(self.mentionKeys[mention]))
			return self.mentionKeys[mention]
		
		for party in allParties:
			keyParty = party.lower().strip()

			#Using distance for full title
			if jf.levenshtein_distance(mention, keyParty) < 5:
				# self.saveMatch('matches.txt', 'Full title jaro distance: ' + str(mention) + ' -> ' + str(party))
				return party

			#Using distance for single word in party title
			for word in keyParty.split():
				# if jf.jaro_distance(mention, word) < 0.2:
				# 	self.saveMatch('Single word jaro distance: ' + str(mention) + ' -> ' + str(party))
				# 	return party
				if jf.levenshtein_distance(mention, word) < 3:
					# self.saveMatch('matches.txt', 'Single word levenstein distance: ' + str(mention) + ' -> ' + str(party))
					return party

		self.totalMentions -= 1
		return False

	def saveMatch(self, filename, line):
		with open(filename, 'a') as fp:
			fp.write(line + '\n')

	def initMentionsKeys(self):
		file = pd.read_csv('mentionsLabelling.csv', sep=',', header=0)
		for key, party in zip(file['key'].tolist(), file['party'].tolist()):
			self.mentionKeys[key] = party

	def initDescriptions(self):
		file = pd.read_csv('spacyNamedEntities.csv', sep=';', header=0)
		for namedEntity, description in zip(file['namedEntity'].tolist(), file['description'].tolist()):
			self.descriptions[namedEntity] = description
