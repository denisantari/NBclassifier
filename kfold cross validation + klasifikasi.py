
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import warnings
from pandas import DataFrame, read_csv
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
warnings.filterwarnings('ignore')

df = pd.read_csv('D:/SKRIPSI/percobaan/1332data9klas/data_1332_9kelas.csv')
y = df.klasifikasi.values

X = pd.read_csv('D:/SKRIPSI/percobaan/1332data9klas/tfidf1332.csv')


# In[2]:

kf = KFold(len(X), n_folds=10, shuffle=True, random_state=9999)
model_train_index = []
model_test_index = []
model = 0

for k, (index_train, index_test) in enumerate(kf):
    X_train, X_test, y_train, y_test = X.ix[index_train,:], X.ix[index_test,:],y[index_train], y[index_test]
    clf = MultinomialNB(alpha=0.1,  fit_prior=True, class_prior=None).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    f1score = f1_score(y_test, clf.predict(X_test))
    precision = precision_score(y_test, clf.predict(X_test))
    recall = recall_score(y_test, clf.predict(X_test))
    print('Model %d has accuracy %f with | f1score: %f | precision: %f | recall : %f'%(k,score, f1score, precision, recall))
    model_train_index.append(index_train)
    model_test_index.append(index_test)
    model+=1


# In[5]:

temp = df.klasifikasi


# In[ ]:




# In[ ]:



