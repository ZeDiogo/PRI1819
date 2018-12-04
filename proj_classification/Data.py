from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import stop_words
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import autosklearn.classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

class Data:
    def __init__(self, data):
      
        self.data = pd.read_csv(data)
    
    def createClasses(self):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.data.party)
        self.data["Class"]=list(self.le.transform(self.data.party))
        
    def callcleanData(self):
        self.data.text = self.data.text.str.replace("[^a-zA-Z]", " ")
        self.data.text = self.data.text.map(lambda x: cleanData(x))

        # tokenization 
        tokenized_doc = self.data['text'].apply(lambda x: x.split())

        # remove stop-words 
        #tokenized_doc = tokenized_doc.apply(lambda x: [item.lower() for item in x if item not in stop_words])
        for index, word_list in enumerate(tokenized_doc):
            tokenized_doc[index] = [word.lower() for word in word_list if word.lower() not in stopwords.words('english')]

        # de-tokenization 
        detokenized_doc = [] 
        for i in range(len(self.data)): 
            t = ' '.join(tokenized_doc[i]) 
            detokenized_doc.append(t) 

        self.data['text'] = detokenized_doc
    
    
    def createEmbeddings(self, word2vec):
        embeddings = get_word2vec_embeddings(word2vec, self.data)
        list_labels=self.data['Class'].tolist()
        self.X_train_word2vec, self.X_test_word2vec, self.y_train_word2vec, self.y_test_word2vec = train_test_split(embeddings, list_labels, test_size=0.2, random_state=40)
        
    def createPartitions(self):
        self.train_feature, self.test_feature, self.train_class, self.test_class = train_test_split(self.data.text, self.data.Class, test_size=0.20,random_state=1)
        
   
    def createVectors(self, stopwords=False, useidf=False, ngram=(1,1), tokenizer=None):
        if (stopwords==True):
            self.vectorizer = TfidfVectorizer(use_idf=useidf, stop_words='english', ngram_range=ngram, min_df=3, max_df=0.9, tokenizer=tokenizer)
        else:   
            self.vectorizer = TfidfVectorizer(use_idf=useidf, min_df=3, max_df=0.9, ngram_range=ngram, tokenizer=tokenizer)

        self.trainvec = self.vectorizer.fit_transform(self.train_feature)
        self.testvec = self.vectorizer.transform(self.test_feature)
        
        return self.vectorizer
    
    def createClassifier(self, mode):
        # SINGLE CLASSIFIERS
        
        classifiers=["MultinomialNB", "KNeighborsClassifier", "Perceptron", "LinearSVC"]
        #classifiers=["MLPClassifier", "DecisionTree", "RandomForest"]
        opt_classifiers=[]
        scores_array=[]
        
        #Once chosen we get all the measures
        if (mode=="chosen"):
            svc=LinearSVC()
            #svc=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
            svc.fit(self.trainvec, self.train_class)
            self.clf=svc
            predictions=svc.predict(self.testvec)
           
            all_metrics=metrics.classification_report(self.test_class, predictions, target_names=self.data.party.unique())

            return [self.clf,  all_metrics]       
        
        #For each classifier (cls) we build an optimal version: opt_cls
        if (mode=="normal"):
            for cls in classifiers:
                opt_cls=self.optimizeClassifier(cls) 
                opt_classifiers.append(opt_cls)
                scores_array.append(opt_cls[2])

            #We fetch the best classifier amongst the optimal versions built previously
            max_acc=max(scores_array)
            index=scores_array.index(max_acc)
            best_cls=opt_classifiers[index]            
            return best_cls
        
        #Word2Vec Embeddings
        if (mode=="w2v"):
            clf_w2v = LinearSVC()
            clf_w2v.fit(self.X_train_word2vec, self.y_train_word2vec)
            y_predicted_word2vec = clf_w2v.predict(self.X_test_word2vec)
            all_metrics=metrics.classification_report(self.y_test_word2vec, y_predicted_word2vec, target_names=self.data.party.unique())
            
            return [all_metrics]
        
        #AUTOMATIC CLASSIFIER - EMSEMBLE
        if (mode=="auto"):
            classifier = autosklearn.classification.AutoSklearnClassifier()
            classifier.fit(self.trainvec, self.train_class)
            predictions = classifier.predict(self.testvec)
            accuracy_auto=accuracy_score(self.test_class, predictions)
            return accuracy_auto
    
    def predict(self, query):
        query=[query]
        query_transformed=self.vectorizer.transform(query)
        predictions=self.clf.predict(query_transformed)
        predictions_labels=[]
        for p in predictions:
            predictions_labels.append(self.le.inverse_transform(p))
        return predictions_labels

    def optimizeClassifier(self, classifier):
        if (classifier=="MLPClassifier"):
            act_parameters = ['relu', 'logistic']
            nn_scores=[]
            for act in act_parameters:
                nn=MLPClassifier(activation=act)
                nn.fit(self.trainvec, self.train_class)
                predictions=nn.predict(self.testvec)
                acc = accuracy_score(self.test_class, predictions)
                nn_scores.append(acc)
                
            max_acc=max(nn_scores)
            index=nn_scores.index(max_acc)
            best=act_parameters[index]
        
        if (classifier=="DecisionTree"):
            samplesdt_parameters = [1, 5]
            dt_scores=[]
            for sample in samplesdt_parameters:
                dt=DecisionTreeRegressor(min_samples_leaf=sample)
                dt.fit(self.trainvec, self.train_class)
                predictions=dt.predict(self.testvec)
                acc = accuracy_score(self.test_class, predictions.round())
                dt_scores.append(acc)
                
            max_acc=max(dt_scores)
            index=dt_scores.index(max_acc)
            best=samplesdt_parameters[index]
            
        if (classifier=="RandomForest"):
            samplesrf_parameters = [1, 5]
            rf_scores=[]
            for sample in samplesrf_parameters:
                rf=RandomForestClassifier(min_samples_leaf=sample)
                rf.fit(self.trainvec, self.train_class)
                predictions=rf.predict(self.testvec)
                acc = accuracy_score(self.test_class, predictions.round())
                rf_scores.append(acc)
                
            max_acc=max(rf_scores)
            index=rf_scores.index(max_acc)
            best=samplesrf_parameters[index]
                    
        if (classifier=="MultinomialNB"):
            alpha_parameters = [0.1,0.5,1.0]
            mnb_scores=[]
            for alpha in alpha_parameters:
                mnb=MultinomialNB(alpha=alpha)
                mnb.fit(self.trainvec, self.train_class)
                predictions=mnb.predict(self.testvec)
                acc = accuracy_score(self.test_class, predictions)
                mnb_scores.append(acc)
                
            max_acc=max(mnb_scores)
            index=mnb_scores.index(max_acc)
            best=alpha_parameters[index]

        if (classifier=="KNeighborsClassifier"):          
            k_parameters = [3,5,7,9,11]
            knn_scores=[]
            for k in k_parameters:
                knn=KNeighborsClassifier(n_neighbors=k)
                knn.fit(self.trainvec, self.train_class)
                predictions=knn.predict(self.testvec)
                acc = accuracy_score(self.test_class, predictions)
                knn_scores.append(acc)
                
            max_acc=max(knn_scores)
            index=knn_scores.index(max_acc)
            best=k_parameters[index]
        
        if (classifier=="Perceptron"):          
            p_parameters = ['None', 'l2' , 'l1' , 'elasticnet']
            perc_scores=[]
            for p in p_parameters:
                perc=Perceptron(max_iter=1000, tol=1e-3, penalty=p)
                perc.fit(self.trainvec, self.train_class)
                predictions=perc.predict(self.testvec)
                acc = accuracy_score(self.test_class, predictions)
                perc_scores.append(acc)
                
            max_acc=max(perc_scores)
            index=perc_scores.index(max_acc)
            best=p_parameters[index]
            
        if (classifier=="LinearSVC"):
            #s_parameters = ['linear', 'rbf']
            s_parameters=['linear']
            svc_scores=[]
            for s in s_parameters:
                svc=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel=s, max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
                svc.fit(self.trainvec, self.train_class)
                predictions=svc.predict(self.testvec)
                acc = accuracy_score(self.test_class, predictions)
                svc_scores.append(acc)
                
            max_acc=max(svc_scores)
            index=svc_scores.index(max_acc)
            best=s_parameters[index]
       
        opt_cls=[classifier, best, max_acc]
        return  opt_cls

    def get_metrics(self, y_test, y_predicted):  
        # true positives / (true positives+false positives)
        precision = precision_score(y_test, y_predicted, pos_label=None,
                                        average='weighted')             
        # true positives / (true positives + false negatives)
        recall = recall_score(y_test, y_predicted, pos_label=None,
                                  average='weighted')

        # harmonic mean of precision and recall
        f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

        # true positives + true negatives/ total
        accuracy = accuracy_score(y_test, y_predicted)
        return accuracy, precision, recall, f1