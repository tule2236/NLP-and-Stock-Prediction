from function import *
import pickle

target = 'CLASSIFICATION'
#target = 'REGRESSION'


#  test on multiple company
company_list = ['005930.KS','AAPL', 'INTC', 'MSFT', 'ORCL', 'SNE',
                 'TSLA','TDC', 'TXN','FB', 'AMZN', 'QCOM', 'GOOG.O',
                'IBM', 'CVX', 'GE','WMT', 'WFC', 'XOM','T','F']


datasets = combine_multiple_companies(company_list)
cal_pos_neg_label(datasets)

# save_data = open("pickled_algos/Abnormal_Guru_Yahoo_data.pickle","wb")
# pickle.dump(datasets, save_data)
# save_data.close()

# open_data_f = open("pickled_algos/Abnormal_Guru_Yahoo_data.pickle", "rb")
# datasets = pickle.load(open_data_f)
# open_data_f.close()

# X_train, X_test, y_train, y_test = prepareDataForClassification(datasets, 0.4)

X_train, X_test, y_train, y_test = PCA_prepareDataForClassification(datasets,150, 0.4)

if target == 'CLASSIFICATION':
    performClassification(X_train, y_train, X_test, y_test, 'MLP')
    print('')

elif target == 'REGRESSION':
    performRegression(datasets, 0.8)
    print('')

