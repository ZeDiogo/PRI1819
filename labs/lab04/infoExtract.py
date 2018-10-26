#!/usr/bin/python

import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def tagAsPOS(words):
	taggedWords = nltk.pos_tag(words)
	# print taggedWords
	return taggedWords

def tagAsNamedEntities(words):
	res = nltk.ne_chunk(words, binary=True)
	# print res
	return res

def exportToFile(file, content):
	with open(file, 'w') as f:
		print >> f, content
	f.close()

def ex1(files):
	words = []
	for file in files:
		print file
		with open(file, 'r') as f:
			doc = f.read()
			doc = doc.decode('utf-8')
			for sentence in nltk.sent_tokenize(doc):
				for word in nltk.word_tokenize(sentence):
					words.append(word)
		f.close()

	taggedWords = tagAsPOS(words)
	print '\n\n\n\n'
	wordsNamedEntities = tagAsNamedEntities(taggedWords)
	exportToFile('tagAsNamedEntities.txt', wordsNamedEntities)

def ex2():
	train = fetch_20newsgroups(subset='train')
	test = fetch_20newsgroups(subset='test')
	# vectorizer = TfidfVectorizer(use_idf=False, stop_words='english', min_df=3, max_df=0.9)
	# testvec = vectorizer.transform(test.data)
	words = []
	for line in test.data:
		for sentence in nltk.sent_tokenize(line):
			for word in nltk.word_tokenize(sentence):
				words.append(word)
	taggedWords = tagAsPOS(words)
	print '\n\n\n\n'
	wordsNamedEntities = tagAsNamedEntities(taggedWords)
	exportToFile('tagAsNamedEntities.txt', wordsNamedEntities)

def main():
	files = ['foodBlog.txt', 'fashionBlog.txt', 'news.txt']
	# ex1(files)
	ex2()

if __name__ == "__main__":
	main()