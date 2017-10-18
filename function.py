import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.decomposition import PCA,FastICA
import operator
from sklearn.metrics import roc_auc_score
import pandas_datareader.data as web
from sklearn.qda import QDA
import datetime as dt 
import time

# from wordCount_samsung import *

def combine_final_df(symbol):
    '''
    df: Fundamental Info + Yahoo Daily Price + Sentiments
    '''
    # Processing Fundamental Info
    print(symbol)
    guru = 'GuruFocus/' + symbol + '_Guru.csv'
    fundamental = pd.read_csv(guru, encoding = "ISO-8859-1")
    fundamental['Filing Date'] = pd.to_datetime(fundamental['Filing Date'])

    idx = pd.date_range('2011-01-01', '2017-07-15')
    fundamental.set_index('Filing Date', inplace=True)
    fundamental.index= pd.DatetimeIndex(fundamental.index)
    fundamental = fundamental.reindex(idx, method='bfill')

    # Sentiment Scores
    sentiment = 'SentimentNews/Abnormal_' + symbol +'_News.csv'
    news_df = pd.read_csv(sentiment, encoding = "ISO-8859-1")
    news_df = news_df[news_df.Text != "['nannan']" ]

    data= news_df.values
    document = []
    for row in data:
        score = row[2].split(',')
        if score[0][2:5] == 'pos':
            document.append( [row[0], row[1], float(score[1][1:-1]) ])
        elif score[0][2:5] == 'neg':
            document.append( [row[0], row[1], float(score[1][1:-1])* -1.0 ])


    news_df = pd.DataFrame(document, columns=['Date','Text','Score'])
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    del news_df['Text']

    # Stock Price df & Generate Labels
    price = 'News/' + symbol+'_Stocks.csv'
    price_df = pd.read_csv(price, encoding = "ISO-8859-1")
    price_df['Return'] = price_df['Adj Close'].pct_change()

    sp500 = pd.read_csv('GuruFocus/Yahoo_Index_GSPC.csv')
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    sp500['Sp_Return'] = sp500['Adj Close'].pct_change()
    del sp500['Open']
    del sp500['Close']
    del sp500['High']
    del sp500['Low']
    del sp500['Volume']
    del sp500['Adj Close']

    price_df= price_df.set_index('Date').join(sp500.set_index('Date'))
    price_df = price_df.dropna()
    price_df['Difference'] = price_df['Return'] - price_df['Sp_Return']

    # Stock Price + Fundamental Info + Sentiments
    sentiment_stock = news_df.set_index('Date').join(price_df)
    df = sentiment_stock.join(fundamental)
    df = df[pd.notnull(df.index)]
    df = df.dropna()
    del df['Fiscal Period']
    del df['Restated Filing Date']
    df = df.replace(to_replace = ['No Debt'], value = 0)

    df['Target'] = np.nan
    requirement = 0.0
    for i in range(len(df)):
        if df['Difference'].iloc[i] > requirement:
            df['Target'].iloc[i] = 1.0
        elif df['Difference'].iloc[i] <  requirement:
            df['Target'].iloc[i] = -1.0
        else:
            df['Target'].iloc[i] = 0.0
    del df['Sp_Return']

    df.reset_index(level=0, inplace=True)
    if 'index' in df.columns:
        df = df.rename( columns = {'index': 'Date'} )
    df = df.dropna(subset = ['Date'])
    df['Date'] = pd.to_datetime( df['Date'])
    del df['Difference']

    return df

def abnormal_df(symbol):
    '''
    df: Fundamental Info + Yahoo Daily Price + Sentiments
    '''
    # Processing Fundamental Info
    print(symbol)
    guru = 'GuruFocus/' + symbol + '_Guru.csv'
    fundamental = pd.read_csv(guru, encoding = "ISO-8859-1")
    fundamental['Filing Date'] = pd.to_datetime(fundamental['Filing Date'])

    idx = pd.date_range('2011-01-01', '2017-07-15')
    fundamental.set_index('Filing Date', inplace=True)
    fundamental.index= pd.DatetimeIndex(fundamental.index)
    fundamental = fundamental.reindex(idx, method='bfill')

    # Sentiment Scores
    sentiment = 'SentimentNews/Abnormal_' + symbol +'_News.csv'
    news_df = pd.read_csv(sentiment, encoding = "ISO-8859-1")
    news_df = news_df[news_df.Text != "['nannan']" ]
    data= news_df.values
    document = []
    for row in data:
        score = row[2].split(',')
        if score[0][2:5] == 'pos':
            document.append( [row[0], row[1], float(score[1][1:-1]) ])
        elif score[0][2:5] == 'neg':
            document.append( [row[0], row[1], float(score[1][1:-1])* -1.0 ])
    news_df = pd.DataFrame(document, columns=['Date','Text','Score'])
    news_df['Date'] = pd.to_datetime(news_df['Date'])
    del news_df['Text']

    # Stock Price df & Generate Labels
    price = 'News/' + symbol+'_Stocks.csv'
    price_df = pd.read_csv(price, encoding = "ISO-8859-1")
    price_df['Return'] = price_df['Adj Close'].pct_change()

    # Abnormal Return
    abnormal_file = 'Abnormal_Returns/' + symbol + '_AbnormalReturn.csv'
    abnormal_df = pd.read_csv(abnormal_file, encoding = "ISO-8859-1")
    abnormal_df['Date'] = pd.to_datetime(abnormal_df['Date'])

    # Stock Price + Fundamental Info + Sentiments
    sentiment_price= news_df.set_index('Date').join(price_df.set_index('Date'))
    sentiment_price_abnormal = sentiment_price.join(abnormal_df.set_index('Date'))
    # df = sentiment_price_abnormal
    df = sentiment_price_abnormal.join(fundamental)
    df = df[pd.notnull(df.index)]
    df = df.dropna()
    del df['Fiscal Period']
    del df['Restated Filing Date']
    df = df.replace(to_replace = ['No Debt'], value = 0)

    # df.fillna(0, inplace = True)
    df['Target'] = np.nan
    requirement = 0.00000
    for i in range(len(df)):
        if df['Abnormal Return'].iloc[i] > requirement:
            df['Target'].iloc[i] = 1.0
        elif df['Abnormal Return'].iloc[i] <  -requirement:
            df['Target'].iloc[i] = -1.0
        else:
            df['Target'].iloc[i] = 0.0

    df.reset_index(level=0, inplace=True)
    if 'index' in df.columns:
        df = df.rename( columns = {'index': 'Date'} )
    df = df.dropna(subset = ['Date'])
    df['Date'] = pd.to_datetime( df['Date'])
    del df['Abnormal Return']
    return df  

def df_no_sentiment(symbol):
    '''
    df: Fundamental Info + Yahoo Daily Price
    '''
    # Processing Fundamental Info
    print(symbol)
    guru = 'GuruFocus/' + symbol + '_Guru.csv'
    fundamental = pd.read_csv(guru, encoding = "ISO-8859-1")
    fundamental['Filing Date'] = pd.to_datetime(fundamental['Filing Date'])

    idx = pd.date_range('2011-01-01', '2017-07-15')
    fundamental.set_index('Filing Date', inplace=True)
    fundamental.index= pd.DatetimeIndex(fundamental.index)
    fundamental = fundamental.reindex(idx, method='bfill')

    # Stock Price df & Generate Labels
    price = 'News/' + symbol+'_Stocks.csv'
    price_df = pd.read_csv(price, encoding = "ISO-8859-1")
    price_df['Return'] = price_df['Adj Close'].pct_change()

    sp500 = pd.read_csv('GuruFocus/Yahoo_Index_GSPC.csv')
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    sp500['Sp_Return'] = sp500['Adj Close'].pct_change()
    del sp500['Open']
    del sp500['Close']
    del sp500['High']
    del sp500['Low']
    del sp500['Volume']
    del sp500['Adj Close']

    price_df= price_df.set_index('Date').join(sp500.set_index('Date'))
    price_df = price_df.dropna()
    price_df['Difference'] = price_df['Return'] - price_df['Sp_Return']

    # Stock Price + Fundamental Info
    df = price_df.join(fundamental)
    df = df[pd.notnull(df.index)]
    df = df.dropna()
    del df['Fiscal Period']
    del df['Restated Filing Date']
    df = df.replace(to_replace = ['No Debt'], value = 0)

    df['Target'] = np.nan
    requirement = 0.0
    for i in range(len(df)):
        if df['Difference'].iloc[i] > requirement:
            df['Target'].iloc[i] = 1.0
        elif df['Difference'].iloc[i] <  requirement:
            df['Target'].iloc[i] = -1.0
        else:
            df['Target'].iloc[i] = 0.0
    del df['Sp_Return']
    # del df['Difference']

    df.reset_index(level=0, inplace=True)
    if 'index' in df.columns:
        df = df.rename( columns = {'index': 'Date'} )
    df = df.dropna(subset = ['Date'])
    df['Date'] = pd.to_datetime( df['Date'])
    del df['Difference']

    return df

def df_no_fundamental(symbol):
    '''
    df: Yahoo Daily Price
    '''
    # Stock Price df & Generate Labels
    print(symbol)
    price = 'News/' + symbol+'_Stocks.csv'
    price_df = pd.read_csv(price, encoding = "ISO-8859-1")
    price_df['Return'] = price_df['Adj Close'].pct_change()

    sp500 = pd.read_csv('GuruFocus/Yahoo_Index_GSPC.csv')
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    sp500['Sp_Return'] = sp500['Adj Close'].pct_change()
    del sp500['Open']
    del sp500['Close']
    del sp500['High']
    del sp500['Low']
    del sp500['Volume']
    del sp500['Adj Close']

    price_df= price_df.set_index('Date').join(sp500.set_index('Date'))
    price_df = price_df.dropna()
    price_df['Difference'] = price_df['Return'] - price_df['Sp_Return']

    df = price_df 

    df['Target'] = np.nan
    requirement = 0.0
    for i in range(len(df)):
        if df['Difference'].iloc[i] > requirement:
            df['Target'].iloc[i] = 1.0
        elif df['Difference'].iloc[i] <  requirement:
            df['Target'].iloc[i] = -1.0
        else:
            df['Target'].iloc[i] = 0.0

    del df['Sp_Return']
    del df['Difference']
    df.reset_index(level=0, inplace=True)
    if 'index' in df.columns:
        df = df.rename( columns = {'index': 'Date'} )
    df = df.dropna(subset = ['Date'])
    df['Date'] = pd.to_datetime( df['Date'])

    return df

def combine_multiple_companies(company_list):
    from sklearn.utils import shuffle

    multiple_dfs = []

    for company in company_list:
        df = combine_final_df(company)
        multiple_dfs.append(df)

    # total = pd.DataFrame()
    # for company in company_list:
    #     total = total.append(df_no_sentiment(company), ignore_index)

    total = pd.concat(multiple_dfs, ignore_index = True)
    total = total.set_index('Date')
    total = total.interpolate(method = 'time')
    total = total.fillna(total.mean())
    total =  shuffle(total)
    return total

def prepareDataForClassification(df, test_size): 
    X = np.array(df.drop(['Target'], 1))
    X = preprocessing.scale(X)
    y = np.array(df['Target'])

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= test_size)
    return X_train, X_test, y_train, y_test

def PCA_prepareDataForClassification(df, n_component, test_size):
    X = np.array(df.drop(['Target'], 1))
    X = preprocessing.scale(X)
    y = np.array(df['Target'])
    
    pca = FastICA(n_components = n_component).fit(X)
    transformed_X = pca.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(transformed_X,y, test_size= test_size)
    
    # var = pca.explained_variance_ratio_
    # print(sum(var))
    # n = list( range(1, n_component+1) )
    # plt.xlabel('Principal Components')
    # plt.ylabel('Percent of Variance Explained')
    # plt.bar(n, var)
    # plt.show()
    return X_train, X_test, y_train, y_test
   

def performTimeSeriesSearchGrid(X_train, y_train, folds, method, grid):
    """
    parameters is a dictionary with: keys --> parameter , values --> list of values of parameter
    """
    print('')
    print('Performing Search Grid CV...')
    print('Algorithm: ', method)
    param = grid.keys()
    finalGrid = {}
    if len(param) == 1:
        for value_0 in grid[param[0]]:
            parameters = [value_0]
            accuracy = performCV(dataset, folds, split, features, method, parameters)
            finalGrid[accuracy] = parameters
        final = sorted(finalGrid.iteritems(), key=operator.itemgetter(0), reverse=True)  
        print('')
        print('Final CV Results: ', final)      
        return final[0]
        
    elif len(param) == 2:
        for value_0 in grid[param[0]]:
            for value_1 in grid[param[1]]:
                parameters = [value_0, value_1]
                accuracy = performCV(dataset, folds, split, features, method, parameters)
                finalGrid[accuracy] = parameters
        final = sorted(finalGrid.iteritems(), key=operator.itemgetter(0), reverse=True)
        print('')
        print('Final CV Results: ', final)
        return final[0]

###############################################################################
######## CLASSIFICATION    
      

def prepareDataForModelSelection(X_train, y_train, start_validation):
    """
    gets train set and generates a validation set splitting the train.
    The validation set is mandatory for feature and model selection.
    """
    X = X_train[X_train.index < start_validation]
    y = y_train[y_train.index < start_validation]    
    
    X_val = X_train[X_train.index >= start_validation]    
    y_val = y_train[y_train.index >= start_validation]   
    
    return X, y, X_val, y_val
  
def performClassification(X_train, y_train, X_test, y_test, method):
    """
    performs classification on returns using serveral algorithms
    """
    #print ''
    print('Performing ' + method + ' Classification...')   
    print('Size of train set: ', X_train.shape)
    print('Size of test set: ', X_test.shape)
   
    if method == 'RF':   
        return performRFClass(X_train, y_train, X_test, y_test)
        
    elif method == 'KNN':
        return performKNNClass(X_train, y_train, X_test, y_test)
    
    elif method == 'SVM':   
        return performSVMClass(X_train, y_train, X_test, y_test)
    
    # elif method == 'ADA':
    #     return performAdaBoostClass(X_train, y_train, X_test, y_test, parameters)
    
    elif method == 'GTB': 
        return performGTBClass(X_train, y_train, X_test, y_test)

    elif method == 'QDA': 
        return performQDAClass(X_train, y_train, X_test, y_test)

    elif method == 'MLP': 
        return performMLPClass(X_train, y_train, X_test, y_test)
    
def performRFClass(X_train, y_train, X_test, y_test):
    """
    Random Forest Binary Classification
    """
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    train_time = end-start
    accuracy = clf.score(X_test, y_test)
 
    y_pred_train = clf.predict(X_train)  
    y_pred_test = clf.predict(X_test)
   

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print('accuracy: ', accuracy,'train mse: ' ,train_mse, 'test mse: ' ,test_mse )
    print('train time: ', train_time)
    return accuracy, train_mse, test_mse
        
def performKNNClass(X_train, y_train, X_test, y_test):
    """
    KNN binary Classification
    """
    clf = neighbors.KNeighborsClassifier()

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start
    accuracy = clf.score(X_test, y_test)

    start = time.time()
    y_pred_train = clf.predict(X_train)
    end = time.time()
    train_time = end-start

    start = time.time()
    y_pred_test = clf.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print('accuracy: ', accuracy,'train mse: ' ,train_mse, 'test mse: ' , test_mse )
    print('fit time: ', fit_time, 'train time: ', train_time, 'test time: ', test_time)
    return accuracy, train_mse, test_mse

def performSVMClass(X_train, y_train, X_test, y_test):
    """
    SVM binary Classification
    """
    clf = SVC()

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start
    accuracy = clf.score(X_test, y_test)

    start = time.time()
    y_pred_train = clf.predict(X_train)
    end = time.time()
    train_time = end-start

    start = time.time()
    y_pred_test = clf.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print('accuracy: ', accuracy,'train mse: ' ,train_mse, 'test mse: ', test_mse )
    print('fit time: ', fit_time, 'train time: ', train_time, 'test time: ', test_time)
    return accuracy, train_mse, test_mse

    
def performAdaBoostClass(X_train, y_train, X_test, y_test, parameters):
    """
    Ada Boosting binary Classification
    """
    n = parameters[0]
    l =  parameters[1]
    clf = AdaBoostClassifier(n_estimators = n, learning_rate = l)

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start
    accuracy = clf.score(X_test, y_test)

    start = time.time()
    y_pred_train = clf.predict(X_train)
    end = time.time()
    train_time = end-start

    start = time.time()
    y_pred_test = clf.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print('accuracy: ', accuracy,'train mse: ' ,train_mse, 'test mse: ' ,test_mse )
    print('fit time: ', fit_time, 'train time: ', train_time, 'test time: ', test_time)
    return accuracy, train_mse, test_mse


    
def performGTBClass(X_train, y_train, X_test, y_test):
    """
    Gradient Tree Boosting binary Classification
    """
    clf = GradientBoostingClassifier(n_estimators=100)

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start
    accuracy = clf.score(X_test, y_test)

    start = time.time()
    y_pred_train = clf.predict(X_train)
    end = time.time()
    train_time = end-start

    start = time.time()
    y_pred_test = clf.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print('accuracy: ', accuracy,'train mse: ' ,train_mse, 'test mse: ' ,test_mse )
    print('fit time: ', fit_time, 'train time: ', train_time, 'test time: ', test_time)
    return accuracy, train_mse, test_mse

def performQDAClass(X_train, y_train, X_test, y_test):
    """
    Gradient Tree Boosting binary Classification
    """
    clf = QDA()

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start
    accuracy = clf.score(X_test, y_test)

    start = time.time()
    y_pred_train = clf.predict(X_train)
    end = time.time()
    train_time = end-start

    start = time.time()
    y_pred_test = clf.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print('accuracy: ', accuracy,'train mse: ' ,train_mse, 'test mse: ', test_mse )
    print('fit time: ', fit_time, 'train time: ', train_time, 'test time: ', test_time)
    return accuracy, train_mse, test_mse

def performMLPClass(X_train, y_train, X_test, y_test):
    clf = MLPClassifier()

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start
    accuracy = clf.score(X_test, y_test)

    start = time.time()
    y_pred_train = clf.predict(X_train)
    end = time.time()
    train_time = end-start

    start = time.time()
    y_pred_test = clf.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print('accuracy: ', accuracy,'train mse: ' ,train_mse, 'test mse: ' ,test_mse )
    print('fit time: ', fit_time, 'train time: ', train_time, 'test time: ', test_time)
    return accuracy, train_mse, test_mse

##############################################################################
####### REGRESSION
    
def performRegression(dataset, split):
    """
    performs regression on returns using serveral algorithms
    """

    features = dataset.columns[1:]
    index = int(np.floor(dataset.shape[0]*split))
    train, test = dataset[:index], dataset[index:]
    print('Size of train set: ', train.shape)
    print('Size of test set: ', test.shape)
    
    output = 'Return_SP500'

    print('Accuracy RFC: ', performRFReg(train, test, features, output))
   
    print('Accuracy SVM: ', performSVMReg(train, test, features, output))
   
    print('Accuracy BAG: ', performBaggingReg(train, test, features, output))
   
    print('Accuracy ADA: ', performAdaBoostReg(train, test, features, output))
   
    print('Accuracy BOO: ', performGradBoostReg(train, test, features, output))

    print('Accuracy KNN: ', performKNNReg(train, test, features, output))


def performRFReg(train, test, features, output):
    """
    Random Forest Regression
    """

    forest = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    forest = forest.fit(train[features], train[output])
    Predicted = forest.predict(test[features])
    

    plt.plot(test[output])
    plt.plot(Predicted, color='red')
    plt.show()        
    
    return mean_squared_error(test[output], Predicted), r2_score(test[output], Predicted)

def performSVMReg(train, test, features, output):
    """
    SVM Regression
    """

    clf = SVR()
    clf.fit(train[features], train[output])
    Predicted = clf.predict(test[features])
    
    plt.plot(test[output])
    plt.plot(Predicted, color='red')
    plt.show()        
    
    return mean_squared_error(test[output],Predicted), r2_score(test[output], Predicted)
    
def performBaggingReg(train, test, features, output):
    """
    Bagging Regression
    """
  
    clf = BaggingRegressor()
    clf.fit(train[features], train[output])
    Predicted = clf.predict(test[features])
    
    plt.plot(test[output])
    plt.plot(Predicted, color='red')
    plt.show()        
    
    return mean_squared_error(test[output],Predicted), r2_score(test[output], Predicted)  

def performAdaBoostReg(train, test, features, output):
    """
    Ada Boost Regression
    """

    clf = AdaBoostRegressor()
    clf.fit(train[features], train[output])
    Predicted = clf.predict(test[features])
    
    plt.plot(test[output])
    plt.plot(Predicted, color='red')
    plt.show()        
    
    return mean_squared_error(test[output],Predicted), r2_score(test[output], Predicted)

def performGradBoostReg(train, test, features, output):
    """
    Gradient Boosting Regression
    """
    
    clf = GradientBoostingRegressor()
    clf.fit(test[features], train[output])
    Predicted = clf.predict(test[features])
    
    plt.plot(test[output])
    plt.plot(Predicted, color='red')
    plt.show()    
    
    return mean_squared_error(test[output],Predicted), r2_score(test[output], Predicted)

def performKNNReg(train, test, features, output):
    """
    KNN Regression
    """

    clf = KNeighborsRegressor()
    clf.fit(train[features], train[output])
    Predicted = clf.predict(test[features])
    
    plt.plot(test[output])
    plt.plot(Predicted, color='red')
    plt.show()        
    
    return mean_squared_error(test[output],Predicted), r2_score(test[output], Predicted)

def cal_pos_neg_label(df):
    print( "Total number of Instances: ", len(df) )

    pos = df[df['Target'] == 1.0] 
    print( "Total number of Positive Labels: ", len(pos) ) # 629 instances

    neg = df[df['Target'] == -1.0]
    print( "Total number of Positive Labels: ", len(neg) ) # 624 instances