# import modules necessary for all the following functions
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import pandas_datareader.data as web
import datetime as dt 

# start = dt.datetime(2012,12,31)
# end = dt.datetime(2017,7,2)
# stock_df = web.DataReader("005930.KS","yahoo",start,end)

def readJson(filename):
    """
    reads a json file and returns a clean pandas data frame
    """
    import pandas as pd
    df = pd.read_csv(filename,  encoding = "ISO-8859-1")
    
    stock_df['Date'] = stock_df['Date'].apply(lambda x: x[:10])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors= 'coerce')
    
    df['Text'] = df['Title'] + df['Body'] 
 
    df = df.drop('Body', 1)
    df = df.drop('Title', 1)

    return df

def cleanText(text):
    """
    removes punctuation, stopwords and returns lowercase text in a list of single words
    """
    text = word_tokenize(text)
    
    from nltk.corpus import stopwords
    clean = [word for word in text if word not in stopwords.words('english')] 
    return clean

def loadPositive():
    """
    loading positive dictionary
    """
    myfile = open('Pos_Neg_Dictionary/LoughranMcDonald_Positive.csv', "r")
    positives = myfile.readlines()
    positive = [pos.strip().lower() for pos in positives]
    return positive

def loadNegative():
    """
    loading positive dictionary
    """
    myfile = open('Pos_Neg_Dictionary/LoughranMcDonald_Negative.csv', "r")
    negatives = myfile.readlines()
    negative = [neg.strip().lower() for neg in negatives]
    return negative

def countNeg(cleantext, negative):
    """
    counts negative words in cleantext
    """
    negs = [word for word in cleantext if word in negative]
    return len(negs)

def countPos(cleantext, positive):
    """
    counts negative words in cleantext
    """
    lst =[]

    for word in cleantext:
        if word in positive:
            lst.append(word)

    return len(lst) 

def getSentiment(cleantext, negative, positive):
    """
    counts negative and positive words in cleantext and returns a score accordingly
    """
    positive = loadPositive()
    negative = loadNegative()
    return (countPos(cleantext, positive) - countNeg(cleantext, negative))

def updateSentimentDataFrame(df):
    """
    performs sentiment analysis on single text entry of dataframe and returns dataframe with scores
    """  
    positive = loadPositive()
    negative = loadNegative()   
    # x: cleantext
    df['Text'] = df['Text']
    df['Score'] = df['Text'].apply(lambda x: getSentiment(x,negative, positive))
    # df['Score'] = getSentiment(x,negative, positive)
    # print(df)
    return df

# df = pd.read_csv('combined.csv', index_col = 'Date',  encoding = "ISO-8859-1")
# df = readJson('export.csv')
# updateSentimentDataFrame(df)

def prepareToConcat(filename):
    """
    load a csv file and gets a score for the day
    """
    df = pd.read_csv(filename, parse_dates=['date'])
    df = df.drop('text', 1)
    df = df.dropna()
    df = df.groupby(['date']).mean()
    name = re.search( r'/(\w+).csv', filename)
    df.columns.values[0] = name.group(1)
    return df



def mergeSentimenToStocks(stock_df):
    news_df = generate_sentiment(file_name)
    # stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    # df= pd.merge(news_df, stock_df, left_index=True, right_index=True)

    # df = pd.read_csv('combined.csv', index_col = 'Date',  encoding = "ISO-8859-1")
    final = stock_df.join(news_df, how='left')
    # print(final)
    return final

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

def generate_sentiment(df_name):
    
    news_df = pd.read_csv(df_name,  encoding = "ISO-8859-1")
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors= 'coerce')
    news_df['Text'] = news_df['Body'].astype(str).str.cat(news_df['Title'].astype(str))
    del news_df['Body']
    del news_df['Title']
    news_df.fillna('', inplace=True)

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
  # news_df = generate_sentiment(name)
  
  scored_name = 'SentimentNews/WordCount_' + company + '_News.csv'
  news_df.to_csv(scored_name)
  print(news_df.head())

# mergeSentimenToStocks(stock_df)


