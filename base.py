import pandas as pd
import abc
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from preprocessor import Preprocessor
import pickle


class SupervisedAlgorithm(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._model = None
        self._vocab = None

    @abc.abstractmethod
    def predict(self, models):
        pass

    @abc.abstractmethod
    def transform_to_tfidf(self, contents):
        pass

    @abc.abstractmethod
    def update_model(self, contents, labels):
        pass
    

class RandomForestAlgorithm(SupervisedAlgorithm):
    def __init__(self):
        super(RandomForestAlgorithm, self).__init__()
        self._model = None
        self._vocab = None

    def predict(self, contents):
        self._model, self._vocab = pickle.load(open('model/model.pkl'))
        tfidf = self.transform_to_tfidf(contents)
        predict = self._model.predict(tfidf)
        return predict

    def transform_to_tfidf(self, contents, get_vocab=False):
        vector = (
            CountVectorizer(analyzer='word', decode_error='replace',
                            vocabulary=self._vocab)
            if not get_vocab else (
            CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.01, max_df=.9,
                            max_features=1000, decode_error='replace')
            )
        )
        vectorized = vector.fit_transform(contents)
        return (
            TfidfTransformer(norm='l2').fit_transform(vectorized).toarray()
            if not get_vocab else
            vector.vocabulary_, TfidfTransformer(norm='l2').fit_transform(vectorized).toarray()
        )

    def update_model(self, contents, labels):
        self._vocab, tfidf = self.transform_to_tfidf(contents, get_vocab=True)
        self._model = (RandomForestClassifier(n_jobs=-1, criterion='entropy')
                       .fit(contents, labels))
        pickle.dump([self._model, self._vocab], open('mode/model.pkl'))

class KFold(object):
    pass



#
# def get_vectorized(_contents):
#     vector = CountVectorizer(analyzer='word', min_df=0.01, max_df=.9, max_features=1000, ngram_range=(1,3),
#                              decode_error='replace')
#     vector_train = vector.fit_transform(_contents)
#     transformer = TfidfTransformer(norm='l2')
#     return transformer.fit_transform(vector_train).toarray()
#
#
# class AlgorithmSupervised(object):
#     metaclass = abc.ABCMeta
#
#     def __init__(self, _x):
#         self._X = _x
#
#     def random_forest(self, label):
#         x_train, x_test, y_train, y_test = train_test_split(self._X, label, test_size=.7)
#         _rfc = RandomForestClassifier(n_jobs=-1, criterion='entropy')
#         _rfc.fit(x_train, y_train)
#         return accuracy_score(y_test, _rfc.predict(x_test))
#
#     @abc.abstractmethod
#     def nb_classifier(self):
#         pass
#
# if __name__ == '__main__':
#     df = pd.read_csv('stemming1332.csv')
#     contents = df.hasil_stemming.tolist()
#     y = pd.read_csv('data_1332_9kelas.csv').klasifikasi.tolist()
#     contents = Preprocessor(contents).run()
#     algorithm = AlgorithmSupervised(get_vectorized(contents))
#     print 'accuracy with random forest : {}'.format(algorithm.random_forest(y))

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
