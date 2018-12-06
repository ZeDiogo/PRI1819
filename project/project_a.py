import pandas as pd
import numpy as np
import os.path
import shutil
import operator
import sys
from whoosh.fields import *
from whoosh.index import create_in, open_dir
from whoosh.query import Variations
from whoosh.qparser import MultifieldParser, OrGroup, QueryParser, MultifieldPlugin
from whoosh.analysis import StemmingAnalyzer
from whoosh.formats import Frequency
from whoosh import scoring
from whoosh.reading import IndexReader


#que informar interessante manifestos ? -> site
#ordem score -> variaçao de stemming (reduzir a palavra dá mais peso a esse tema geral!)

#query -> ver o top 5 docs se faz sentido ordem
#frase_query ->

def main():

    #create DataFrame for file with dataset with columns: text, manifesto_id, party, date, title
    data = processData("en_docs_clean.csv")
    index = createIndex(data)
    #reader = index.reader()
    data.to_csv('processed_data.csv', sep='\t', encoding='utf-8')
    query = input("Introduce query: ")
    #print(query)
    number_docs = input("How many manifestos do you wish to be presented (introduce an integer): ")
    #print("Number of arguments:", sys.argv[2])
    for arg in sys.argv:
        if arg == 'BM25F':
            print("Using BM25F as scoring criteria with common default values B=0.75 and k1=1.5 ")
            w = scoring.BM25F(B=0.75, content_B=1.0, K1=1.5)
            searchQuery(query, index, w, number_docs)
        elif arg == 'TF-IDF':
            print("Using TF-IDF as scoring criteria")
            w = scoring.TF_IDF()
            searchQuery(query, index, w, number_docs)
        elif arg == 'Frequency':
            print("Using Frequency as scoring criteria")
            w = scoring.Frequency()
            #check how it does frequency -> sum?? !!!!!1
            searchQuery(query, index, w, number_docs)

    #they will have different orders! of displaying the docs

    #By default, Whoosh returns the results ordered using the BM25 similarity.
    #Consider not only the term frequency and inverse document
    #frequency heuristics, but also the document length as a
    #normalization factor for the term frequency

    #statistics
    response = input("To know the number of manifestos per party write yes: ")
    if response == 'yes':
        number_manifestos_party(data)

    response = input("To know the number of times each keyword of the query introduced before is mentioned by each party write yes: ")
    if response == 'yes':
        keyword_times_party(index, query)
    response = input("Type theme/keyword for which you which to know the year it was most relevant: ")
    if response:
        year_relevant_keyword(index, response)

def processData(FILE):
    data = pd.read_csv(FILE)
    dic = {}

    for i in range(data.shape[0]):
        key = data.loc[i, "manifesto_id"]
        if key not in dic:
            dic[key] = []
            #append text
            dic[key].append(data.loc[i,"text"])
            #manifesto_id
            dic[key].append(key)
            #party
            #dic[key][2] = data.loc[i,"party"]
            dic[key].append(data.loc[i,"party"])
            #data
            dic[key].append(data.loc[i,"date"])
            #title
            dic[key].append(data.loc[i,"title"])
            #print("this should be 5 but is:", len(dic[key]))
        else:
            dic[key][0] += data.loc[i,"text"]
            #for each manifesto, same party, date and title

    #create DataFrame from dictionary, with row using dict keys, allowing to merge every text for the same manifesto
    data_proc = pd.DataFrame.from_dict(dic, orient='index')
    #print(data_proc.shape[0])
    return data_proc

def createIndex(data):
    #RegexTokenizer + LowerCaseFilter + StopWordsFilter + Stemming filter
    analyzer = StemmingAnalyzer()
    vector_format = Frequency() #Stores the number of times each term appears in each document.
    schema = Schema(text=TEXT(analyzer=analyzer, vector=vector_format), manifesto_id=ID(stored=True, sortable=True), party=TEXT(stored=True, vector=vector_format, sortable=True), date=NUMERIC(stored=True), title=TEXT(stored=True, analyzer=analyzer, vector=vector_format))

    if os.path.isdir("index"):
        shutil.rmtree("index")

    if not os.path.exists("index"):
        os.mkdir("index")

    index = create_in("index", schema)

    #create an index writer to add documents
    writer = index.writer()

    for ind,row in data.iterrows():
        writer.add_document(text=row[0], manifesto_id=row[1], party=row[2], date=row[3], title=row[4])

    writer.commit()

    return index

# Relevant manifestos for a given query composed of keywords, ordered by relevance/Score
def searchQuery(arg, index, w, n):
    ix = open_dir("index")
    with ix.searcher(weighting = w) as searcher:

        #search() takes query object and returns result object
        query_parser = QueryParser(None, schema=ix.schema, group=OrGroup, termclass=Variations)
        query_parser.add_plugin(MultifieldPlugin(["title","text"]))
        # OrGroup -> so that any of the terms may be present for a document to match

        #stopwords will be removed from the query since the fields where defined in the schema to ignore stopwords, this applies for lowercase and stemming.

        #query.Variations -> searches for morphological variations of the given word in the same field

        query = query_parser.parse(arg)
        results = searcher.search(query, limit=int(n))
        #By default the results contains at most the first 10 matching documents; limit=None all results;
        print ("Number of results:", results.scored_length())
        docs = []
        for result in results:
            print("Score:", result.score)
            print("Manifesto:", result['manifesto_id'])
            docs.append(result['manifesto_id'])
        print("Documents that match with query ordered by score:", docs)

#definition of BM25 scoring



#Number of parties
def number_parties(data):
    num = data[2].nunique()
    print("There are %s parties" % num)

#Number of manifestos per party
def number_manifestos_party(data):
    #party corresponds to the third column of the DataFrame
    num = data[2].value_counts()
    num_manif = data[0].nunique()
    print("\nFor %s manifestos, we have the following distribution per party:" % (num_manif))
    print(num)

#How many times each party mentions each keyword
def keyword_times_party(index, arg):
    ix = open_dir("index")
    with index.searcher(weighting = scoring.Frequency()) as searcher:
        query_parser = QueryParser(None, schema=ix.schema, group=OrGroup, termclass=Variations)
        query_parser.add_plugin(MultifieldPlugin(["title","text"]))

        number_times = {}
        list_keywords = process_keywords(arg)
        print("the Keywords are:", list_keywords)
        for word in list_keywords:
            number_times[word] = []
            #dictionary to store number of times word appears for a certain party
            #key = party value=number_times
            query = query_parser.parse(word)
            results = searcher.search(query, limit=None)
            dic_aux = {}
            for result in results:
                if result['party'] not in dic_aux:
                    dic_aux[result['party']] = result.score
                else:
                    dic_aux[result['party']] += result.score
            #we have dic_aux with frequency for word for each party as a total for all results
            for key,value in dic_aux.items():
                number_times[word].append([key,value])
        print_dict_party(number_times)

def process_keywords(arg):
    analyzer = StemmingAnalyzer()
    list_keywords = [token.text for token in analyzer(arg)]
    print("two", list_keywords)
    return list_keywords


def print_dict_party(dictionary):
    for key, value in dictionary.items():
        print("\nNumber of times per party for keyword:", key)
        for i in range(len(value)):
            print("%s - %s" % (value[i][0], int(value[i][1])))

# What was the year more relevant for a give subject/keyword? (i.e., keyword appears more times)
def year_relevant_keyword(index, arg):
    ix = open_dir("index")
    with index.searcher(weighting = scoring.Frequency()) as searcher:
        query_parser = QueryParser(None, schema=ix.schema, termclass=Variations)
        query_parser.add_plugin(MultifieldPlugin(["title","text"]))
        number_times = {}
        #dictionary to store number of times query matches for a certain date
        query = query_parser.parse(arg)
        results = searcher.search(query, limit=None)
        #only the manifestos that contain all terms of query (AND, not OR)
        for result in results:
            print(result)
            print("Score:", result.score)
            year = int(result['date']/100)
            if year not in number_times:
                number_times[year] = result.score
            else:
                number_times[year] += result.score
        get_year(number_times)

def get_year(number_times, arg):
    year_max = max(number_times.items(), key=operator.itemgetter(1))[0]
    for key,value in number_times.items():
        print("Data: %s Number_times: %s" % (key,value))

    print("Most relevant year for the theme '%s': %s", (arg,year_max))







if __name__ == "__main__":
	main()
