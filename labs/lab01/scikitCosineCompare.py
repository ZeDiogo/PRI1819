from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scikitCountWords import scikitCountWords


def scikitCosineCompare(file1, file2):
	dictionary = {}

	with open(file1) as f1:
		with open(file2) as f2:
			# Merge common words from both files into vocabulary
			array = []
			array.append(f1.read())
			array.append(f2.read())
			vect = TfidfVectorizer()
			X=vect.fit_transform(array)
			vocabulary = vect.get_feature_names()

			# creation of independent vectors for both files containing 
			# frequency of words
			freq1 = np.zeros(len(vocabulary))
			freq2 = np.zeros(len(vocabulary))
			file1Counts = scikitCountWords(file1)
			file2Counts = scikitCountWords(file2)
			i = 0
			for word in vocabulary:
				if file1Counts.has_key(word):
					freq1[i] = file1Counts[word]
				if file2Counts.has_key(word):
					freq2[i] = file2Counts[word]
				i += 1

			similarity = np.dot(freq1, freq2) / np.sqrt(np.dot(freq1, freq1) * np.dot(freq2, freq2))

			print 'Similarity: ' + str(similarity * 100) + '%'

	f1.close()
	f2.close()

def main():
	scikitCosineCompare("text.txt", "text2.txt")

if __name__ == "__main__":
	main()