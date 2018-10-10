from occurrences import occurrencesDict
import sys

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

	print 'Statistics:'
	print 'Total number of...'
	print 'Docs: {}'.format(numberDocs+1)
	print 'Terms: {}'.format(numberTerms)
	print 'Individual terms:'
	i = 0
	for word, lst in occurrences.items():
		print '{}: {}'.format(word, numberIndividualTerms[i])
		i += 1

	parseArgs()

def parseArgs():
	for arg in sys.argv:
		if arg == 'DF':
			# Document frequency
			df()
		elif arg == 'TF':
			# Maximum and minimum term frequency
			tf()
		elif arg == 'IDF':
			# Inverse Document Frequency
			idf()

def df():
	print 'DF'

def tf():
	print 'TF'
	
def idf():
	print 'IDF'

def main():
	statistics("text.txt")

if __name__ == "__main__":
	main()