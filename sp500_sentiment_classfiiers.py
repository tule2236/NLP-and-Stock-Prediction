import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
# import pandas_datareader.data as web
# import datetime as dt
import nltk
import numpy as np
import random
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
import pickle 
import pdb
from nltk.stem import WordNetLemmatizer
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.stem import PorterStemmer
import pandas_datareader.data as web
import datetime as dt
import math
# Create Word Dict (list with lenth 1644 days)
lemmatizer = WordNetLemmatizer()
# ps = PorterStemmer()

def create_news_df(news_file):
    news_df = pd.read_csv(news_file, encoding = "ISO-8859-1")
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors= 'coerce')
    news_df['Text'] = news_df['Body'].astype(str).str.cat(news_df['Title'].astype(str))
    del news_df['Body']
    del news_df['Title']
    news_df.fillna('', inplace=True)
    return news_df

def create_stock_df(symbol):
    """
    downloads stock which is gonna be the output of prediciton
    """
    name = 'News/'+symbol + '_Stocks.csv'
    out =   pd.read_csv(name, encoding = "ISO-8859-1")
    # out['Date'] = pd.to_datetime(out['Date'])
    out['Return'] = out['Adj Close'].pct_change()

    sp500 = pd.read_csv('GuruFocus/Yahoo_Index_GSPC.csv')
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    sp500['Sp_Return'] = sp500['Adj Close'].pct_change()

    del sp500['Open']
    del sp500['Close']
    del sp500['High']
    del sp500['Low']
    del sp500['Volume']
    del sp500['Adj Close']

    df= out.set_index('Date').join(sp500.set_index('Date'))
    df = df.dropna()

    df['Difference'] = df['Return'] - df['Sp_Return']
    return df

def combine_final_df(news_df, stock_df):
    df= stock_df.join(news_df.set_index('Date'))
    df.fillna(0, inplace = True)
    df['Target'] = np.nan
    requirement = 0.0
    for i in range(len(df)):
        if df['Difference'].iloc[i] > requirement:
            df['Target'].iloc[i] = 1.0
        elif df['Difference'].iloc[i] <  requirement:
            df['Target'].iloc[i] = -1.0
        else:
            df['Target'].iloc[i] = 0.0
    return df

def combine_multiple_company(company_list):  
    """
    Concatenate multiple dataframe of companies together to create big lexicon
    """
    company_dfs = []

    for company in company_list:
        print(company)
        news_file_name = 'News/'+ company + '_News.csv'
        news = create_news_df(news_file_name)
        
        stock = create_stock_df(company)

        final = combine_final_df(news, stock)
        company_dfs.append(final)
    total = pd.concat(company_dfs, ignore_index = True)
    return total

company_list = ["005930.KS", 'AAPL', 'INTC', 'MSFT', 'ORCL', 'SNE',
                'TDC', 'TSLA', 'TXN', 'FB', 'AMZN', 'QCOM', 'GOOG.O',
                'IBM', 'CVX', 'GE', 'VZ','WMT', 'WFC', 'XOM','T','F']

# company_list = ["005930.KS", 'AAPL']              

news_df = combine_multiple_company(company_list)

def tag_label_to_event(news_df):
    data = news_df.values
    document = []
    for row in data: 
        if row[-1] == 1.0 and row[-2] != 'nannan':
            document.append( (row[-2], "pos") )
        elif row[-1] == -1.0 and row[-2] != 'nannan':
            document.append( (row[-2], "neg") )
    return document

def process_news_data(words):
    """
    Cleaning News data
    """
    lst = []
    for s in words:
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)
        sentence_token = word_tokenize(s.lower())
        nostopword_sentence = []
        for word_token in sentence_token:

            stemmed_word = lemmatizer.lemmatize(word_token)
            # stemmed_word = ps.stem(word_token)
            if stemmed_word not in stopwords.words('english'):
                nostopword_sentence.append(stemmed_word)
            # if word_token not in stopwords.words('english'):
            #     nostopword_sentence.append(word_token)
        lst.append(nostopword_sentence)
    return lst
    
def create_lexicon(news_df):
    text = np.array(process_news_data(news_df['Text'].values))
    all_words = []
    allowed_word_types = ['J','V','R']
    for day in list(range(len(text))):
        words = text[day]
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
    all_words = nltk.FreqDist(all_words)
    word_features = []
    for i in all_words.keys():
        if all_words.get(i) > 2:
            word_features.append(i)
    # word_features.remove('nannan')
    return word_features

def bigram_word_features(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams_tuple = bigram_finder.nbest(score_fn, n)
    bigrams = [' '.join(each) for each in bigrams_tuple]
    return list([ngram for ngram in itertools.chain(words, bigrams)])


singular_word_features = create_lexicon(news_df)
word_features = bigram_word_features(singular_word_features)
# word_features = create_lexicon(news_df)
print('number of word features', len(word_features))

document = tag_label_to_event(news_df)

save_document = open("pickled_algos/documents_22_sp500.pickle","wb")
pickle.dump(document, save_document)
save_document.close()

save_word_features = open("pickled_algos/word_features_22_sp500.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(line):
    sentence = word_tokenize(line)
    # words = sentence
    words = []
    for each in sentence:
        words.append( lemmatizer.lemmatize(each) ) 
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [ (find_features(line),category) for (line,category) in document]
random.shuffle(featuresets)
print('number of featuresets', len(featuresets))

save_features = open("pickled_algos/featuresets_22_sp500.pickle","wb")
pickle.dump(featuresets, save_features)
save_features.close()

# featuresets_f = open("pickled_algos/featuresets_22_sp500.pickle", "rb")
# featuresets = pickle.load(featuresets_f)
# featuresets_f.close()

num = math.ceil(len(featuresets)*0.80)
training_set = featuresets[:num]
testing_set = featuresets[num:]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features(20)
save_classifier = open("pickled_algos/sp500_originalnaivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
save_classifier = open("pickled_algos/sp500_MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
save_classifier = open("pickled_algos/sp500_BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression(dual = True))
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
save_classifier = open("pickled_algos/sp500_LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

Weighted_LogisticRegression_classifier = SklearnClassifier(LogisticRegression(class_weight = 'balanced'))
Weighted_LogisticRegression_classifier.train(training_set)
print("Weighted_LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(Weighted_LogisticRegression_classifier, testing_set))*100)
save_classifier = open("pickled_algos/sp500_Weighted_LogisticRegression_classifier.pickle","wb")
pickle.dump(Weighted_LogisticRegression_classifier, save_classifier)
save_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
save_classifier = open("pickled_algos/sp500_SGD_classifier.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/sp500_LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

PolySVC_classifier = SklearnClassifier(SVC(kernel = 'poly', C= 0.01))
PolySVC_classifier.train(training_set)
print("PolySVC_classifier accuracy percent:", (nltk.classify.accuracy(PolySVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/sp500_PolySVC_classifier.pickle","wb")
pickle.dump(PolySVC_classifier, save_classifier)
save_classifier.close()

RbfSVC_classifier = SklearnClassifier(SVC(C= 0.01))
RbfSVC_classifier.train(training_set)
print("RbfSVC_classifier accuracy percent:", (nltk.classify.accuracy(RbfSVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/sp500_RbfSVC_classifier.pickle","wb")
pickle.dump(RbfSVC_classifier, save_classifier)
save_classifier.close()

NuSVC_classifier = SklearnClassifier(NuSVC(kernel = 'poly'))
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/sp500_NuSVC_classifier.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

MLP_classifier = SklearnClassifier(MLPClassifier(hidden_layer_sizes=(100,100,100,100)))
MLP_classifier.train(training_set)
print("MLP_classifier accuracy percent:", (nltk.classify.accuracy(MLP_classifier, testing_set))*100)
save_classifier = open("pickled_algos/sp500_MLP_classifier.pickle","wb")
pickle.dump(MLP_classifier, save_classifier)
save_classifier.close()

Big_MLP_classifier = SklearnClassifier(MLPClassifier(hidden_layer_sizes=(300,300,300,300,300)))
Big_MLP_classifier.train(training_set)
print("Big_MLP_classifier accuracy percent:", (nltk.classify.accuracy(Big_MLP_classifier, testing_set))*100)
save_classifier = open("pickled_algos/sp500_Big_MLP_classifier.pickle","wb")
pickle.dump(Big_MLP_classifier, save_classifier)
save_classifier.close()

# pdb.set_trace()





