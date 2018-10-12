from occurrences import occurrencesDict
import sys
import math

def statistics(FILE):
	occurrences = occurrencesDict(FILE)

	numberIndividualTerms = []
	numberTerms = 0
	numberDocs = 0
	i = 0
	for vocab, occur in occurrences.items():
		numberTerms += 1
		numberIndividualTerms.append(0)
		for oc in occur:
			if oc[0] > numberDocs:
				numberDocs = oc[0]
			numberIndividualTerms[i] += oc[1]
		i += 1

	numberDocs += 1
	print 'Statistics:'
	print 'Total number of...'
	print 'Docs: {}'.format(numberDocs)
	print 'Terms: {}'.format(numberTerms)
	print 'Individual terms:'
	i = 0
	for word, lst in occurrences.items():
		print '{}: {}'.format(word, numberIndividualTerms[i])
		i += 1

	parseArgs(occurrences, numberDocs)

def parseArgs(occurrences, numberDocs):
	for arg in sys.argv:
		if arg == 'DF':
			# Document frequency
			df(occurrences)
		elif arg == 'TF':
			# Maximum and minimum term frequency
			tf(occurrences, numberDocs)
		elif arg == 'IDF':
			# Inverse Document Frequency
			idf(occurrences, numberDocs)

def df(occurrences):
	print '##############Document Frequency:'
	dfs = []
	for vocab, occur in occurrences.items():
		dfs.append(len(occur))
		print '{}: {}'.format(vocab, len(occur))
	return dfs

def df(occurrences, terms):
	dfs = df(occurrences)
	dfTerms = []

def tf(occurrences, numberDocs):
	print 'Maximum and minimum term frequency:'
	for vocab, occur in occurrences.items():
		max = 0
		min = 9999
		for t in occur:
			if max < t[1]:
				max = t[1]
			if min > t[1]:
				min = t[1]
		if len(occur) < numberDocs:
			min = 0
		print '{}: Max: {} | Min: {}'.format(vocab, max, min)

	
def idf(occurrences, numberDocs):
	print 'Inverse Document Frequency'
	dfs = df(occurrences)
	vocab = occurrences.keys()
	idfs = {}
	for i in range(0, len(occurrences)):
		idf = math.log10(numberDocs/dfs[i])
		idfs[vocab[i]].append(idf)
		print '{}: idf: {}'.format(vocab[i], idf)
	return idfs

def main():
	statistics("text.txt")

if __name__ == "__main__":
	main()