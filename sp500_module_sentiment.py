import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing

from nltk.corpus import stopwords
import string
import pandas_datareader.data as web
import datetime as dt 


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents_f = open("pickled_algos/documents_22_abnormalReturn.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open("pickled_algos/word_features_22_abnormalReturn.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()



open_file = open("pickled_algos/sp500_originalnaivebayes.pickle", "rb")
originalnaivebayes = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/sp500_MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/sp500_BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/sp500_LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/sp500_SGD_classifier.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/sp500_LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/sp500_NuSVC_classifier.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/sp500_PolySVC_classifier.pickle","rb")
PolySVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/sp500_RbfSVC_classifier.pickle","rb")
RbfSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/sp500_Weighted_LogisticRegression_classifier.pickle","rb")
Weighted_LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/sp500_MLP_classifier.pickle", "rb")
MLP_classifier = pickle.load(open_file)
open_file.close()

# open_file = open("pickled_algos/sp500_Big_MLP_classifier.pickle","rb")
# Big_MLP_classifier = pickle.load(open_file)
# open_file.close()

voted_classifier = VoteClassifier(
                                  originalnaivebayes,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDC_classifier,                             
                                  LinearSVC_classifier,
                                  NuSVC_classifier,
                                  PolySVC_classifier,
                                  RbfSVC_classifier,
                                  Weighted_LogisticRegression_classifier,
                                  MLP_classifier)
                                  # Big_MLP_classifier)

def find_features(document):
    words = document
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

def getSentiment(text):
    feats = find_features(text)
    print(1)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

def processing(words):
    lst = []
    lemmatizer = WordNetLemmatizer()
    for s in words:
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)
        sentence_token = word_tokenize(s.lower())
        nostopword_sentence = []
        for word_token in sentence_token:
            stemmed_word = lemmatizer.lemmatize(word_token)
            if stemmed_word not in stopwords.words('english'):
                nostopword_sentence.append(stemmed_word)
        lst.append(nostopword_sentence)
    return lst

def updateSentimentDataFrame(df):
    """
    performs sentiment analysis on single text entry of dataframe and returns dataframe with scores
    """  
    df['Text'] = df['Text']
    df['Score'] = df['Text'].apply(lambda x: getSentiment(x))
    return df

def generate_sentiment(df_name):
    
    news_df = pd.read_csv(df_name,  encoding = "ISO-8859-1")
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors= 'coerce')
    # news_df.fillna('', inplace=True)
    news_df.dropna()
    news_df['Text'] = news_df['Body'].astype(str).str.cat(news_df['Title'].astype(str))
    
    del news_df['Body']
    del news_df['Title']
    

    text = np.array(processing(news_df['Text'].values))
    text = pd.DataFrame(text, index = news_df['Date'])
    text.columns = ['Text']
    news_df = updateSentimentDataFrame(text)

    return news_df

company_list = ["005930.KS", 'AAPL', 'INTC', 'MSFT', 'ORCL', 'SNE',
                'TDC', 'TSLA', 'TXN', 'FB', 'AMZN', 'QCOM', 'GOOG.O',
                'IBM', 'CVX', 'GE','WMT', 'WFC', 'XOM','T','F']

company_dfs = []
for company in company_list:
  print('print',company)
  name = 'News/' + company + '_News.csv'
  news_df = generate_sentiment(name)
  
  scored_name = 'SentimentNews/Sp500_' + company + '_News.csv'
  news_df.to_csv(scored_name)
  print(news_df.head())

# print(getSentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))

# print(getSentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))

# print(getSentiment('Tech group sides with Apple in Qualcomms iPhone ban dispute'))

# print(getSentiment('Qualcomms profit forecast disappoints as Apple battle takes toll'))

# print(getSentiment('Four Apple contractors accuse Qualcomm of antitrust violations'))

# print(getSentiment('Patent lawyer for Apple dies after battle with cancer'))

# print(getSentiment('Apple set to expand Siri, taking different route from Amazons Alexa'))

# print(getSentiment('Chipmakers at Taiwans biggest tech fair look beyond crowded smartphone market'))




