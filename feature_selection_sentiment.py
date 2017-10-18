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

# Create Word Dict (list with lenth 1644 days)
# lemmatizer = WordNetLemmatizer()
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
        news_file_name = 'News/'+ company + '_News.csv'
        news = create_news_df(news_file_name)
        stock_file_name = 'Abnormal_Returns/' + company + '_AbnormalReturn.csv'
        stock = create_stock_df(stock_file_name)

        final = combine_final_df(news, stock)
        company_dfs.append(final)
    total = pd.concat(company_dfs, ignore_index = True)
    return total

company_list = ['Samsung', 'Apple', 'Intel', 'Microsoft', 'Oracle', 'Sony',
                'Teradata', 'Tesla', 'TexasInstruments', 'Facebook', 'Amazon']
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

def create_word_scores():
    """
    Combine words and bigrams and compute words and bigrams information scores
    """
    posSentences = open('polarityData\\rt-polaritydata\\rt-polarity-pos.txt', 'r')
    negSentences = open('polarityData\\rt-polaritydata\\rt-polarity-neg.txt', 'r')
    posSentences = re.split(r'\n', posSentences.read())
    negSentences = re.split(r'\n', negSentences.read())
 
    #creates lists of all positive and negative words
    posWords = []
    negWords = []
    for i in posSentences:
        posWord = re.findall(r"[\w']+|[.,!?;]", i)
        posWords.append(posWord)
    for i in negSentences:
        negWord = re.findall(r"[\w']+|[.,!?;]", i)
        negWords.append(negWord)
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 8000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 8000)

    pos = posWords + posBigrams
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd.inc(word)
        cond_word_fd['pos'].inc(word)
    for word in negWords:
        word_fd.inc(word)
        cond_word_fd['neg'].inc(word)

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

def find_best_words(word_scores, number):
    """
    Second we should extact the most informative words or bigrams based on the information score
    """
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

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

            # stemmed_word = lemmatizer.lemmatize(word_token)
            # stemmed_word = ps.stem(word_token)
            # if stemmed_word not in stopwords.words('english'):
            #     nostopword_sentence.append(stemmed_word)
            if word_token not in stopwords.words('english'):
                nostopword_sentence.append(word_token)
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

save_document = open("pickled_algos/documents.pickle","wb")
pickle.dump(document, save_document)
save_document.close()


def find_features(line):
    sentence = word_tokenize(line)
    words = sentence
    # words = []
    # for each in sentence:
    #     words.append( ps.stem(each) ) 
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Transform review to features by setting labels to words in review
def pos_features(feature_extraction_method):
    posFeatures = []
    for i in pos:
        posWords = [feature_extraction_method(i),'pos']
        posFeatures.append(posWords)
    return posFeatures

def neg_features(feature_extraction_method):
    negFeatures = []
    for j in neg:
        negWords = [feature_extraction_method(j),'neg']
        negFeatures.append(negWords)
    return negFeatures


best_words = find_best_words(word_scores, 1500) # Set dimension and initiallize most informative words
posFeatures = pos_features(best_word_features)
negFeatures = neg_features(best_word_features)

dimention = ['500','1000','1500','2000','2500','3000']

for d in dimention:
    word_scores = create_word_bigram_scores()
    best_words = find_best_words(word_scores, int(d))

    posFeatures = pos_features(best_word_features_com)
    negFeatures = neg_features(best_word_features_com)

    # Make the feature set ramdon
    shuffle(posFeatures)
    shuffle(negFeatures)

    # 75% of features used as training set (in fact, it have a better way by using cross validation function)
    size_pos = int(len(pos_review) * 0.75)
    size_neg = int(len(neg_review) * 0.75)

    trainset = posFeatures[:size_pos] + negFeatures[:size_neg]
    testset = posFeatures[size_pos:] + negFeatures[size_neg:]

    test, tag_test = zip(*testset)

    print 'BernoulliNB`s accuracy is %f' %score(BernoulliNB())
    print 'MultinomiaNB`s accuracy is %f' %score(MultinomialNB())
    print 'LogisticRegression`s accuracy is %f' %score(LogisticRegression())
    print 'SVC`s accuracy is %f' %score(SVC())
    print 'LinearSVC`s accuracy is %f' %score(LinearSVC())
    print 'NuSVC`s accuracy is %f' %score(NuSVC())
    print 



featuresets = [ (find_features(line),category) for (line,category) in document]
random.shuffle(featuresets)
print(len(featuresets))

save_features = open("pickled_algos/featuresets.pickle","wb")
pickle.dump(featuresets, save_features)
save_features.close()

featuresets_f = open("pickled_algos/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()


training_set = featuresets[:4500]
testing_set = featuresets[4500:]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(20)
save_classifier = open("pickled_algos/originalnaivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
save_classifier = open("pickled_algos/MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB(alpha=3.0))
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
save_classifier = open("pickled_algos/BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression(dual = True))
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
save_classifier = open("pickled_algos/LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

Weighted_LogisticRegression_classifier = SklearnClassifier(LogisticRegression(class_weight = 'balanced'))
Weighted_LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(Weighted_LogisticRegression_classifier, testing_set))*100)
save_classifier = open("pickled_algos/Weighted_LogisticRegression_classifier.pickle","wb")
pickle.dump(Weighted_LogisticRegression_classifier, save_classifier)
save_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier(loss='log'))
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
save_classifier = open("pickled_algos/SGD_classifier.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC(C= 0.01))
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

PolySVC_classifier = SklearnClassifier(SVC(kernel = 'poly', C= 0.01))
PolySVC_classifier.train(training_set)
print("PolySVC_classifier accuracy percent:", (nltk.classify.accuracy(PolySVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/PolySVC_classifier.pickle","wb")
pickle.dump(PolySVC_classifier, save_classifier)
save_classifier.close()

RbfSVC_classifier = SklearnClassifier(SVC(C= 0.01))
RbfSVC_classifier.train(training_set)
print("RbfSVC_classifier accuracy percent:", (nltk.classify.accuracy(RbfSVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/RbfSVC_classifier.pickle","wb")
pickle.dump(RbfSVC_classifier, save_classifier)
save_classifier.close()

NuSVC_classifier = SklearnClassifier(NuSVC(kernel = 'poly'))
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
save_classifier = open("pickled_algos/NuSVC_classifier.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

MLP_classifier = SklearnClassifier(MLPClassifier(hidden_layer_sizes=(100,100,100,100)))
MLP_classifier.train(training_set)
print("MLP_classifier accuracy percent:", (nltk.classify.accuracy(MLP_classifier, testing_set))*100)
save_classifier = open("pickled_algos/NuSVC_classifier.pickle","wb")
pickle.dump(MLP_classifier, save_classifier)
save_classifier.close()

pdb.set_trace()





