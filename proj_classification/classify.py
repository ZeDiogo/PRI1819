from Data import Data

from nltk.stem.snowball import SnowballStemmer
import nltk
import spacy
from nltk.stem import WordNetLemmatizer
import string
import re
from string import punctuation
import gensim

#implement tokenizers taken as input in vectorizer
def tokenizer1(doc):
    spacy.load('en')
    lemmatizer = spacy.lang.en.English()
    tokens = lemmatizer(doc)
    doc=" ".join([token.lemma_ for token in tokens])
    return doc

def tokenizer2(doc):
    wordnet_lemmatizer = WordNetLemmatizer()
    doc = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in nltk.word_tokenize(doc)])
    return doc

def tokenizer3(doc):
    english_stemmer = SnowballStemmer("english")
    doc = " ".join([english_stemmer.stem(w) for w in nltk.word_tokenize(doc)])
    return doc

#Define cleanData function called by callcleanData() function of Data object
def cleanData(text):
    txt = str(text)
    
    # Replace apostrophes with standard lexicons
    txt = txt.replace("isn't", "is not")
    txt = txt.replace("aren't", "are not")
    txt = txt.replace("ain't", "am not")
    txt = txt.replace("won't", "will not")
    txt = txt.replace("didn't", "did not")
    txt = txt.replace("shan't", "shall not")
    txt = txt.replace("haven't", "have not")
    txt = txt.replace("hadn't", "had not")
    txt = txt.replace("hasn't", "has not")
    txt = txt.replace("don't", "do not")
    txt = txt.replace("wasn't", "was not")
    txt = txt.replace("weren't", "were not")
    txt = txt.replace("doesn't", "does not")
    txt = re.sub(r"\'ve", " have ", txt)
    txt = re.sub(r"can't", "cannot ", txt)
    txt = re.sub(r"n't", " not ", txt)
    txt = re.sub(r"I'm", "I am", txt)
    txt = re.sub(r" m ", " am ", txt)
    txt = re.sub(r"\'re", " are ", txt)
    txt = re.sub(r"\'d", " would ", txt)
    txt = re.sub(r"\'ll", " will ", txt)
    
    # Remove urls and emails
    txt = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', txt, flags=re.MULTILINE)
    txt = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', txt, flags=re.MULTILINE)
    
    # Remove punctuation from text
    txt = ''.join([c for c in text if c not in punctuation])

    # Remove all symbols
    txt = re.sub(r'[^A-Za-z0-9\s]',r' ',txt)
    txt = re.sub(r'\n',r' ',txt)
    
    txt = re.sub(r'[0-9]',r' ',txt)
    
    return txt

#Define word2vec model taken as argument in method createEmbeddings(word2vec) of Data object
#word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
#word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

#Functions called by createEmbeddigs() method of Data object
def get_average_word2vec(tokens_list, vector, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, data, generate_missing=False):
    embeddings = data['text'].apply(lambda x: get_average_word2vec(x, vectors))
    return list(embeddings)

#Define values to consider in Data object
value1=False
value2=True
ngram=(1,4)

#create Data object
def main():

	file=open("en_docs_clean.csv", "r")
	data = Data(file)
	data.createClasses()
	data.createPartitions()
	vectorizer=data.createVectors(stopwords=value1, useidf=value2, ngram=ngram)
	clf, all_metrics, cm=data.createClassifier("chosen")
	print()
	print("Classifier Trained:")
	print(clf)
	print()
	print("Metrics of trained classifier:")
	print(all_metrics)
	print()
	print("Confusion matrix in the same order:")
	print("(number of observations known to be in group of line i but predicted to be in group of row j)")
	print(cm)
	query=input("Please provide query for prediction: ")
	predictions=data.predict(query)
	print(predictions)

if __name__ == "__main__":
    main()