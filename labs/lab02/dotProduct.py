from occurrences import occurrencesDict

def sim(FILE, terms):
	termsDic = {}
	occur = occurrencesDict(FILE)
	for term in terms:
		if occur.has_key(term):
			termsDic[term] = occurrences[term]
		else:
			termsDic[term] = [[0,0]]

	

def processQuery(queryFile):
	terms = []
	with open(queryFile) as file:
		for line in file:
			for term in line:
				terms.append(term)
	return terms

def main():
	terms = processQuery("query.txt")
	sim("text.txt", terms)

if __name__ == "__main__":
	main()