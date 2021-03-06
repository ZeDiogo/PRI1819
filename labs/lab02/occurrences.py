from sklearn.feature_extraction.text import CountVectorizer
# from prettyPrint import prettyPrint

def printOccur(occurrences):
	print('Vocabulary    Occurences')
	for vocab, occur in occurrences.items():
		print '{} {}'.format(vocab, occur)

def occurrencesDict(FILE):
	occurrences = {}
	docs = []

	with open(FILE) as file:
		cv = CountVectorizer()
		for doc in file:
			docs.append(doc)
		cv_fit=cv.fit_transform(docs)
	file.close()

	vocabulary = cv.get_feature_names()
	print vocabulary
	frequency = cv_fit.toarray()
	for i in range(0, len(vocabulary)):
		occurrences[vocabulary[i]] = []
		for j in range(len(docs)):
			if frequency[j][i] != 0:
				occurrences[vocabulary[i]].append([j, frequency[j][i]])

	# printOccur(occurrences)

	return occurrences


def main():
	occurrencesDict("text.txt")
	

if __name__ == "__main__":
	main()