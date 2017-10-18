import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
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
import math
from nltk.metrics.scores import accuracy, precision, recall
import collections
import os, sys
import time




# Create Word Dict (list with lenth 1644 days)
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def create_news_df(news_file):
    news_df = pd.read_csv(news_file, encoding = "ISO-8859-1")
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors= 'coerce')
    news_df['Text'] = news_df['Body'].astype(str).str.cat(news_df['Title'].astype(str))
    del news_df['Body']
    del news_df['Title']
    news_df.fillna('', inplace=True)
    return news_df

def create_stock_df(stock_file):
    #  append Target into News dataframe
    stock_df = pd.read_csv(stock_file, encoding = "ISO-8859-1")
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    return stock_df

def combine_final_df(news_df, stock_df):
    df= stock_df.set_index('Date').join(news_df.set_index('Date'))
    df.fillna(0, inplace = True)
    df['Target'] = np.nan
    print(df.head())
    requirement = 0.00000
    for i in range(len(df)):
        if df['Abnormal Return'].iloc[i] > requirement:
            df['Target'].iloc[i] = 1.0
        elif df['Abnormal Return'].iloc[i] <  -requirement:
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
        stock_file_name = 'Abnormal_Returns/' + company + '_AbnormalReturn.csv'
        stock = create_stock_df(stock_file_name)

        final = combine_final_df(news, stock)
        company_dfs.append(final)
    total = pd.concat(company_dfs, ignore_index = True)
    return total

# company_list = ["005930.KS", 'AAPL', 'INTC', 'MSFT', 'ORCL', 'SNE',
#                 'TDC', 'TSLA', 'TXN', 'FB', 'AMZN', 'QCOM', 'GOOG.O',
#                 'IBM', 'CVX', 'GE','WMT', 'WFC', 'XOM','T','F']

company_list = ["005930.KS", 'AAPL']
news_df = combine_multiple_company(company_list)
# news_df.to_csv('combined_companies.csv')

# news_df = pd.read_csv('combined_companies.csv')

def tag_label_to_event(news_df):
    data = news_df.values
    document = []
    for row in data: 
        if row[2] == 1.0 and row[1] != 'nannan':
            document.append( (row[1], "pos") )
        elif row[2] == -1.0 and row[1] != 'nannan':
            document.append( (row[1], "neg") )
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
print(len(word_features))

document = tag_label_to_event(news_df)

save_document = open("pickled_algos/documents_22_abnormalReturn.pickle","wb")
pickle.dump(document, save_document)
save_document.close()

save_word_features = open("pickled_algos/word_features_22_abnormalReturn.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(line):
    sentence = word_tokenize(line)
    # words = sentence
    # words = []
    for each in sentence:
        words.append( ps.stem(each) ) 
    for w in word_features:
        features[w] = (w in words)
    return features
    

featuresets = [ (find_features(line),category) for (line,category) in document]


random.shuffle(featuresets)
print(len(featuresets))


save_features = open("pickled_algos/featuresets_22_abnormalReturn.pickle","wb")
pickle.dump(featuresets, save_features)
save_features.close()

featuresets_f = open("pickled_algos/featuresets_22_abnormalReturn.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

posfeats = []
negfeats = []
for each in featuresets:
    if each[1] == "pos":
        posfeats.append(each)
    else:
        negfeats.append(each)

split = 0.6
num = math.ceil(len(featuresets)* split)
training_set = featuresets[:num]
testing_set = featuresets[num:]

negcutoff = math.ceil(len(negfeats)*split)
poscutoff = math.ceil(len(posfeats)*split)

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)



def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
file = open(os.path.join('sentiment_result.csv'), mode='a')
file.write("algorithm, split point, accuracy, pos precision, neg precision, rpos recall, neg recall\n")

algos_list = [MultinomialNB(), SVC()]


for algo in algos_list:
    start = time.time()
    classifier = SklearnClassifier(algo).train(trainfeats)
    end = time.time()
    train_time = end-start

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    accuracy = nltk.classify.accuracy(classifier, testfeats)
    pos_precison = precision(refsets['pos'], testsets['pos'])
    neg_precision = precision(refsets['neg'], testsets['neg'])
    pos_recall = recall(refsets['pos'], testsets['pos'])
    neg_recall = recall(refsets['neg'], testsets['neg'])

    pickle_name = "pickled_algos/" + str(algo) + "_classifier.pickle"
    save_classifier = open(pickle_name,"wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

    algo_name = str(algo)[:-2]

    file.write(str(algo)+","+str(split)+","+str(train_time)+","
            + str(accuracy)+","+str(pos_precison)+","+str(neg_precision)+","
            +str(pos_recall)+","+str(neg_recall)+"\n")

# print('accuracy:', nltk.classify.accuracy(classifier, testfeats) )
# print('pos precision:', precision(refsets['pos'], testsets['pos']) )
# print( 'pos recall:', recall(refsets['pos'], testsets['pos']) )
# print( 'neg precision:', precision(refsets['neg'], testsets['neg']) )
# print( 'neg recall:', recall(refsets['neg'], testsets['neg']) )

# SVC_classifier = SklearnClassifier(SVC()).train(trainfeats)
# save_classifier = open("pickled_algos/SVC_classifier.pickle","wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()
# print('SVC accuracy:', nltk.classify.accuracy(classifier, testfeats) )
# print('SVC pos precision:', precision(refsets['pos'], testsets['pos']) )
# print( 'SVC pos recall:', recall(refsets['pos'], testsets['pos']) )
# print( 'SVC neg precision:', precision(refsets['neg'], testsets['neg']) )
# print( 'SVC neg recall:', recall(refsets['neg'], testsets['neg']) )



# classifier = nltk.NaiveBayesClassifier.train(training_set)
# print("Original Naive Bayes Algo accuracy, pos_accuracy, neg_accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100, 
#     nltk.classify.accuracy(classifier, pos_feature[:poscutoff]), nltk.classify.accuracy(classifier, neg_feature[negcutoff:]))
# classifier.show_most_informative_features(20)
# save_classifier = open("pickled_algos/originalnaivebayes.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
# save_classifier = open("pickled_algos/MNB_classifier.pickle","wb")
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()

# BernoulliNB_classifier = SklearnClassifier(BernoulliNB(alpha=3.0))
# BernoulliNB_classifier.train(training_set)
# print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
# save_classifier = open("pickled_algos/BernoulliNB_classifier.pickle","wb")
# pickle.dump(BernoulliNB_classifier, save_classifier)
# save_classifier.close()

# LogisticRegression_classifier = SklearnClassifier(LogisticRegression(dual = True))
# LogisticRegression_classifier.train(training_set)
# print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
# save_classifier = open("pickled_algos/LogisticRegression_classifier.pickle","wb")
# pickle.dump(LogisticRegression_classifier, save_classifier)
# save_classifier.close()

# Weighted_LogisticRegression_classifier = SklearnClassifier(LogisticRegression(class_weight = 'balanced'))
# Weighted_LogisticRegression_classifier.train(training_set)
# print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(Weighted_LogisticRegression_classifier, testing_set))*100)
# save_classifier = open("pickled_algos/Weighted_LogisticRegression_classifier.pickle","wb")
# pickle.dump(Weighted_LogisticRegression_classifier, save_classifier)
# save_classifier.close()

# SGDClassifier_classifier = SklearnClassifier(SGDClassifier(loss='log'))
# SGDClassifier_classifier.train(training_set)
# print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
# save_classifier = open("pickled_algos/SGD_classifier.pickle","wb")
# pickle.dump(SGDClassifier_classifier, save_classifier)
# save_classifier.close()

# LinearSVC_classifier = SklearnClassifier(LinearSVC(C= 0.01))
# LinearSVC_classifier.train(training_set)
# print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
# save_classifier = open("pickled_algos/LinearSVC_classifier.pickle","wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()

# PolySVC_classifier = SklearnClassifier(SVC(kernel = 'poly', C= 0.01))
# PolySVC_classifier.train(training_set)
# print("PolySVC_classifier accuracy percent:", (nltk.classify.accuracy(PolySVC_classifier, testing_set))*100)
# save_classifier = open("pickled_algos/PolySVC_classifier.pickle","wb")
# pickle.dump(PolySVC_classifier, save_classifier)
# save_classifier.close()

# RbfSVC_classifier = SklearnClassifier(SVC(C= 0.01))
# RbfSVC_classifier.train(training_set)
# print("RbfSVC_classifier accuracy percent:", (nltk.classify.accuracy(RbfSVC_classifier, testing_set))*100)
# save_classifier = open("pickled_algos/RbfSVC_classifier.pickle","wb")
# pickle.dump(RbfSVC_classifier, save_classifier)
# save_classifier.close()

# NuSVC_classifier = SklearnClassifier(NuSVC(kernel = 'poly'))
# NuSVC_classifier.train(training_set)
# print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
# save_classifier = open("pickled_algos/NuSVC_classifier.pickle","wb")
# pickle.dump(NuSVC_classifier, save_classifier)
# save_classifier.close()

# MLP_classifier = SklearnClassifier(MLPClassifier(hidden_layer_sizes=(100,100,100,100)))
# MLP_classifier.train(training_set)
# print("MLP_classifier accuracy percent:", (nltk.classify.accuracy(MLP_classifier, testing_set))*100)
# save_classifier = open("pickled_algos/MLP_classifier.pickle","wb")
# pickle.dump(MLP_classifier, save_classifier)
# save_classifier.close()

# Big_MLP_classifier = SklearnClassifier(MLPClassifier(hidden_layer_sizes=(300,300,300,300,300)))
# Big_MLP_classifier.train(training_set)
# print("Big_MLP_classifier accuracy percent:", (nltk.classify.accuracy(Big_MLP_classifier, testing_set))*100)
# save_classifier = open("pickled_algos/Big_MLP_classifier.pickle","wb")
# pickle.dump(Big_MLP_classifier, save_classifier)
# save_classifier.close()

# pdb.set_trace()





