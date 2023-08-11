## Adapted from Daas & van der Doef (2020)
### Code updated for package updates/soon to be obsolete functions

from collections import defaultdict
import os
import numpy as np
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection 
import sklearn.metrics
import pandas as pd

##Set path
path = "" 
os.chdir(path)
##Set datasets
datasets = {"BusinessReg": "Datasets/2_ABR_adj.csv", # ABR wordsfile (LARGE)
            "BusinessReg_samp": "Datasets/NHR_htmls_40000_sample.csv", # ABR sample
            "Startups": "Datasets/6_Startups_abr_adj.csv", # Startups wordsfile
            "CIS2016": "Datasets/1_InnovatiefSurveySDS.csv", # Original CIS2016 Scrape 1 wordsfile
            "CIS2016_sc1_words_for_fulltexts":"Datasets/wordsfile_matched_urls.csv", # CIS2016 ids for which fulltext available
            "CIS2016_sc5_words": "Datasets/5_InnovatieSurvey2_2b_concp15.csv", # CIS2016 Scrape 5 wordsfile
            "SC2_fulltexts": "Datasets/SCR01_htmls_NHR_in_CIS_and_SCR05.csv", # Fulltext ids of scr 2
            "SC5_fulltexts": "Datasets/SCR05_htmls_EM_in_CIS_and_SCR01.csv"} # Fulltext ids of scr 5

##match fulltext to cis2016 words
CIS2016 = pd.read_csv(datasets['CIS2016'], sep = ";")
CIS2016_sc1_words_for_fulltexts = pd.read_csv(datasets["CIS2016_sc1_words_for_fulltexts"], sep = ";")
CIS2016_sc5_words = pd.read_csv(datasets['CIS2016_sc5_words'], sep = ";")
SC2_fulltexts = pd.read_csv(datasets["SC2_fulltexts"], sep = ",")
SC5_fulltexts = pd.read_csv(datasets['SC5_fulltexts'], sep = ",")

SC2_words = CIS2016_sc1_words_for_fulltexts[['id_NHR', 'Innov', 'text', 'lang']]
SC5_words = pd.merge(SC5_fulltexts, 
                         CIS2016_sc5_words, 
                         on = ['BEID', 'Innov'])

ABR = pd.read_csv(datasets["BusinessReg"], sep = ";")
ABR_samp = pd.read_csv(datasets["BusinessReg_samp"], sep = ",")
ABR_words = pd.merge(ABR_samp, ABR, left_on = "id", right_on = "BEID")
ABR_words.rename(columns = {"Innov_x":"Innov", "text_y":"text"}, inplace = True)


##Read data 6 only once
df6 = pd.read_csv(datasets["Startups"], sep=";")
df6 = df6.fillna(" ")
##algotitme selected
alg = LogisticRegression(penalty='l1', solver='liblinear')
##cval = 10 ##Cross validation amount

def addlanguagefeature(X, dataset):
    taal = dataset['lang'].tolist()
    language_vector = []
    for item in taal:
        if item=="dutch":
            language_vector.append(0)
        else:
            language_vector.append(1)
    language_vector = np.array(language_vector, ndmin=2, dtype="float64").T
    X = np.c_[X, language_vector]
    return(X)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(word2vec))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def combineCSVs(j, s, df6):
    ##Read data ABR once and process only once
    trainset = datasets[j]
    df2 = pd.read_csv(path+trainset, sep=";")
    df2 = df2.fillna(" ")

    ## sample trainset
    df_sample = df2.sample(n=s)   
    ##remove df2 (to prevent memory issues)
    del df2

    ##remove any duplicates from df6 (with same BEID in df_Sample)
    ids = df_sample['BEID'].tolist()
    df6 = df6[~df6['BEID'].isin(ids)]        
    ##combine pandas dataframes
    frames = [df_sample, df6]
    df_sample = pd.concat(frames, sort=True)
    
    return(df_sample)

def getText(df):
    text = []
    for file in df['text'].tolist():
        if isinstance(file, str):
            text.append(file.split())
        else:
            text.append(" ")
    return(text)


def INNClassify(data, WordEmb, nchar = 3, wv = 200, mindf = 100, sampSurv = 20000, n = 1):
    """
    Classify companies as innovative or not based on website texts.
    
    Parameters:
    -----------
    data:      str - "CIS2016" for original model with 
                                   original data (Daas & van der Doef, 2020); 
                     "BusinessReg" for original model with 
                                   ABR sample + startup data  (Daas & van der Doef, 2020);
                     "scr02_sent" for model performance on 
                                  subset contained in sentence embedding model (t = 3);
                     "scr05_sent" for model performance on 
                                  subset contained in sentence embedding model (t = 12);
                     "test_t-0_on_ABR" for original model performance (t = 0) 
                                       on external validation subset from sentence model (t = 3)
    WordEmb:   T/F - add word embeddings?
    nchar:     int - 2 or 3 word characters?    (default: 3)
    wv:        int - wordvector size            (default: 200)
    mindf:     int - min document frequency     (default: 100)
    sampSurv:  int - sample from businesses (default: 20,000)
    n:         int - number of times to run
    -----------
    
    Returns:
        Accuracy
        Precision
        Recall
        F1-score
    """
    
    ##fit model
    for i in range(n):  ##To run this multiple times, each run is an individual fit
        print(i)
        ##Read data ABR once and process only once        
        if data == "CIS2016":
            df_sample = CIS2016
            df_sample = df_sample.fillna(" ")
        elif data == "BusinessReg":
            df_sample = combineCSVs('BusinessReg', sampSurv, df6)
        elif data == "scr02_sent":
            df_sample = SC2_words
            df_sample = df_sample.fillna(" ")
        elif data == "scr05_sent":
            df_sample = SC5_words
            df_sample = df_sample.fillna(" ")
        elif data == "BusinessReg_samp":
            df_sample = ABR_words
            df_sample = df_sample.fillna(" ")
        elif data == "test_t-0_on_ABR":
            df_sample = CIS2016
            df_sample = df_sample.fillna(" ")
            df_test = ABR_words
            df_test = df_test.fillna(" ")

        

        ## tekst voor training set
        text = getText(df_sample)
                
        ## maak en train het word2vec model (duurt lang)
        if WordEmb:
            sizeword2vec = wv
            model = word2vec.Word2Vec(text, vector_size=sizeword2vec)
            w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))
    
        ## Draw sample from training set
        y = np.array(df_sample['Innov'])
        
        if data == "test_t-0_on_ABR":
            X_train = df_sample.drop(columns = 'Innov')
            y_train = df_sample['Innov']
            X_test = df_test.drop(columns = 'Innov')
            y_test = df_test['Innov']
        else:
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df_sample, y, test_size=0.20, random_state=42)
         
        ##get text from training set for wordemb
        if WordEmb:
            textWE = getText(X_train)                
            p = MeanEmbeddingVectorizer(w2v)
            p = p.fit(np.array(textWE, dtype = "object"), y_train)
    
        ## tfidf
        #bigrammen de nrange ngram_range=(1,2)
        if nchar == 2:
            cv = CountVectorizer(input='content', min_df=mindf, token_pattern=u'\w{2,}')
        elif nchar == 3:
            cv = CountVectorizer(input='content', min_df=mindf, token_pattern=u'\w{3,}')
                
        word_count_vector=cv.fit_transform(X_train['text'].tolist())
        #tfidfvectorizer  = sklearn.feature_extraction.text.TfidfVectorizer(input='content', min_df=100, token_pattern=u'\w{2,}')
        tfidfvectorizer = sklearn.feature_extraction.text.TfidfTransformer(smooth_idf=True,use_idf=True,sublinear_tf=True)
        X2 = tfidfvectorizer.fit(word_count_vector)
        feature_names = cv.get_feature_names_out()
        feature_names = feature_names.tolist()
    
        Xtrain2 = tfidfvectorizer.transform(word_count_vector)
    
        ## word2vec
        if WordEmb:
            Xtrain = p.transform(np.array(textWE, dtype = "object"))
            X_train2 = np.c_[Xtrain, Xtrain2.toarray()]
        else:
            X_train2 = np.c_[Xtrain2.toarray()]
    
        X_train3 = addlanguagefeature(X_train2, X_train)
    
        ##alg is selected before loop above
        alg.fit(X_train3, y_train)
                
        if WordEmb:
            f = ["wordembedding"]*sizeword2vec
            f.extend(feature_names)
        else:
            f = feature_names
    
        f.extend(["Feature_taal"])
        ##f.extend(["Innovative_words"])
        feature_names = f
    
        # toepassen op de test data
        text_test = getText(X_test)
                
        word_count_vector_test=cv.transform(X_test['text'].tolist())
        b2 = tfidfvectorizer.transform(word_count_vector_test)
    
        if WordEmb:
            b = p.transform(np.array(text_test, dtype = "object"))
            b = np.c_[b, b2.toarray()]
        else:
            b = np.c_[b2.toarray()]
    
        b = addlanguagefeature(b, X_test)
        
        ##predict model on test set
        ypred = alg.predict(b)
        ##get accuracy
        acc = sklearn.metrics.accuracy_score(y_test, ypred, normalize=True)
        prec = sklearn.metrics.precision_score(y_test, ypred)
        rec = sklearn.metrics.recall_score(y_test, ypred)
        f1 = sklearn.metrics.f1_score(y_test, ypred)
        ##show result
        print("data:", data, "- wordemb", WordEmb, "- nchar:", nchar)
        print("accuracy:", acc)
        print("precision:", prec)
        print("recall:", rec)
        print("f1:", f1)

INNClassify(data = "CIS2016", WordEmb = True, nchar = 3)
INNClassify(data = "BusinessReg", WordEmb = True, nchar = 3)
INNClassify(data = "scr02_sent", WordEmb = True, nchar = 3)
INNClassify(data = "scr05_sent", WordEmb = True, nchar = 3)
INNClassify(data = "BusinessReg_samp", WordEmb = True, nchar = 3)
INNClassify(data = "test_t-0_on_ABR", WordEmb = True, nchar = 3)


