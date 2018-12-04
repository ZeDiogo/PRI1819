from Data import Data

value1=False
value2=True
ngram=(1,4)

def main():

	file=open("en_docs_clean.csv", "r")
	data = Data(file)
	data.createClasses()
	data.createPartitions()
	vectorizer=data.createVectors(stopwords=value1, useidf=value2, ngram=ngram)
	clf, all_metrics=data.createClassifier("chosen")
	print()
	print("Classifier Trained:")
	print(clf)
	print()
	print("Metrics of trained classifier:")
	print(all_metrics)
	query=input("Please provide query for prediction: ")
	predictions=data.predict(query)
	print(predictions)

if __name__ == "__main__":
    main()