from sklearn.feature_extraction.text import CountVectorizer
from prettyPrint import prettyPrint

def scikitCountWords(FILE):
	dictionary = {}

	with open(FILE) as file:
		array = []
		array.append(file.read())
		cv = CountVectorizer()
		cv_fit=cv.fit_transform(array)
		i=0
		count = cv_fit.toarray()
		wordCount = count[0]
		for word in cv.get_feature_names():
			dictionary[word] = wordCount[i]
			i+=1
	file.close()

	#prettyPrint(dictionary)
	return dictionary


def main():
	scikitCountWords("text.txt")
	

if __name__ == "__main__":
	main()