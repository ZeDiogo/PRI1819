from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

FILE = "text.txt"

dictionary = {}

with open(FILE) as file:
	array = []
	array.append(file.read())
	cv = CountVectorizer()
	cv_fit=cv.fit_transform(array)
	i=0
	count = cv_fit.toarray()
	for word in cv.get_feature_names():
		print count + ' <-- ' + str(word)
		i+=1
# print(cv.get_feature_names())
# print(cv_fit.toarray())


# texts=["dog cat fish","dog cat cat","fish bird", 'bird']
# cv = CountVectorizer()
# cv_fit=cv.fit_transform(texts)

# print(cv.get_feature_names())
# print(cv_fit.toarray())