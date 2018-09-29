
from prettyPrint import prettyPrint
from nltkCountWords import nltkCountWords

def commonWords(file1, file2):
	dict1 = nltkCountWords(file1)
	dict2 = nltkCountWords(file2)
	dictionary = {}
	for key in dict1:
		if dict2.has_key(key):
			dictionary[key] = min(dict1[key], dict2[key])
	prettyPrint(dictionary)
	return dictionary

def main():
	commonWords("text.txt", "text2.txt")

if __name__ == "__main__":
	main()