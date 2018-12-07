Information Processing and Retrieval - Course Project - Group 8
Andreia Rogério 78557 Francisca Ribeiro 81117 José Diogo Oliveira 75255

a) Ad hoc search on the collection of documents

To run the code for this part in the command line just introduce:

    python project_a.py **scoring**

where the scoring is the method chosen by the user to present the results.
The possibilities for **scoring** include:

- 'BM25'
- 'Cosine'
- 'TF-IDF'

After this, the user must introduce the query in the command line and follow several
intructions, including the number of results to be presented and the three statistics
implemented.

b) To use the classifier run on terminal: 
>> python classify.py

It will display the metrics of the classifier being used and will ask as input a query to be predicted. 

Required libraries:
>> python -m spacy download en

Important notes:
- To use the createClassifier() in "w2v" mode you should download the file GoogleNews-vectors-negative300.bin.gz and un-comment the word2vec initialization lines in classify.py

c) 