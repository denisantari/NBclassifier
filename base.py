
import pandas as pd
import abc
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from preprocessor import Preprocessor, tf_idf


class AlgorithmSupervised(object):
    metaclass = abc.ABCMeta

    def __init__(self, _x):
        self._X = _x

    def random_forest(self, label):
        x_train, x_test, y_train, y_test = train_test_split(self._X, label, test_size=.7)
        _rfc = RandomForestClassifier(n_jobs=-1, criterion='entropy')
        _rfc.fit(x_train, y_train)
        return accuracy_score(y_test, _rfc.predict(x_test))

    @abc.abstractmethod
    def nb_classifier(self):
        pass

if __name__ == '__main__':
    df = pd.read_csv('data_1332_9kelas.csv')
    contents = df.content.tolist()
    y = df.klasifikasi.tolist()
    contents = Preprocessor(contents).run()
    algorithm = AlgorithmSupervised(tf_idf(contents))
    print 'accuracy with random forest : {}'.format(algorithm.random_forest(y))

# kf = KFold(len(X), n_folds=10, shuffle=True, random_state=9999)
# model_train_index = []
# model_test_index = []
# model = 0
#
# for k, (index_train, index_test) in enumerate(kf):
#     X_train, X_test, y_train, y_test = X.ix[index_train,:], X.ix[index_test,:],y[index_train], y[index_test]
#     clf = MultinomialNB(alpha=0.1,  fit_prior=True, class_prior=None).fit(X_train, y_train)
#     score = clf.score(X_test, y_test)
#     f1score = f1_score(y_test, clf.predict(X_test))
#     precision = precision_score(y_test, clf.predict(X_test))
#     recall = recall_score(y_test, clf.predict(X_test))
#     print('Model %d has accuracy %f with | f1score: %f | precision: %f | recall : %f'%(k,score, f1score, precision, recall))
#     model_train_index.append(index_train)
#     model_test_index.append(index_test)
#     model+=1
#
# temp = df.klasifikasi
