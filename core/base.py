import abc
import pickle

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from core.preprocessor import Preprocessor


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

    @abc.abstractmethod
    def fit_model_and_predict(self, x, labels):
        pass


class RandomForestAlgorithm(SupervisedAlgorithm):
    def __init__(self):
        super(RandomForestAlgorithm, self).__init__()
        self._model = None
        self._vocab = None

    def predict(self, contents):
        _contents = Preprocessor().run(contents)
        self._model, self._vocab = pickle.load(
            open('core/model/model-random-forest.pkl', 'rb'))
        tfidf = self.transform_to_tfidf(_contents)
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
        result = (
            TfidfTransformer(norm='l2').fit_transform(vectorized).toarray()
            if not get_vocab else (
                vector.vocabulary_, TfidfTransformer(norm='l2').fit_transform(vectorized).toarray()
            )
        )
        return result

    def update_model(self, contents, labels):
        _contents = Preprocessor().run(contents)
        self._vocab, tfidf = self.transform_to_tfidf(_contents, get_vocab=True)
        self._model = (RandomForestClassifier(n_jobs=-1, criterion='entropy')
                       .fit(_contents, labels))
        pickle.dump([self._model, self._vocab],
                    open('core/model/model-random-forest.pkl', 'wb'))

    def fit_model_and_predict(self, x, labels):
        self._model = RandomForestClassifier(n_jobs=-1, criterion='entropy').fit(x, labels)
        return self._model


class NaiveBayesClassifier(SupervisedAlgorithm):
    def __init__(self):
        super(NaiveBayesClassifier, self).__init__()
        self._model = None
        self._vocab = None

    def predict(self, models):
        pass

    def transform_to_tfidf(self, contents):
        pass

    def update_model(self, contents, labels):
        pass

    def fit_model_and_predict(self, x, labels):
        pass


class KFoldCrossValidation(object):
    def __init__(self):
        self._model = None
        self._vocab = None

    @staticmethod
    def find_best_model(scores):
        return max(scores)

    def run(self, contents, labels):
        _contents = Preprocessor().run(contents)
        self._vocab, X = RandomForestAlgorithm().transform_to_tfidf(_contents, get_vocab=True)
        kfold = KFold(len(X), n_folds=10, shuffle=True, random_state=9999)

        scores = []

        for k, (index_train, index_test) in enumerate(kfold):

            X_train, X_test, y_train, y_test = \
                X[index_train], X[index_test], labels[index_train], labels[index_test]

            model = RandomForestAlgorithm().fit_model_and_predict(X_train, y_train)

            scores.append([model.score(X_test, y_test), model])

        best_scores = self.find_best_model(scores)
        self._model = best_scores[1]
        print 'Best score : {}'.format(best_scores[0])
        pickle.dump([self._model, self._vocab],
                    open('core/model/model-random-forest.pkl', 'wb'))

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
