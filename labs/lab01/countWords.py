
import re
from prettyPrint import prettyPrint

def countWords(FILE):

	dictionary = {}
	pattern = re.compile('[\W]*') #Non alpha-numerics

	with open(FILE) as file:
		for word in file.read().split():
			word = pattern.sub('', word)
			if dictionary.has_key(word):
				dictionary[word] += 1
			else:
				dictionary[word] = 1
	file.close()
	#prettyPrint(dictionary)
	return dictionary

def main():
	countWords("text.txt")
	

if __name__ == "__main__":
	main()
