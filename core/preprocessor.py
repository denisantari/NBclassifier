import abc  #library abstract base class, untuk asbtraksi kelas
import nltk
import os
import string
import time

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

absolute_path = os.path.dirname(os.path.abspath(__file__))


class Preprocessor(object):
    """
    staticmethod digunakan untuk menggabungkan fungsi yg memupunyai beberapa logical
    connection with a class to the classs, disini maksudnya fungsi stemmer,
    punctuation dan stopword adalah fungsi yg saling berkaitan
    """

    __metaclass__ = abc.ABCMeta

    PUNCTUATION = string.punctuation
    STOPWORDS = [line.rstrip('\n') for line in open(
        absolute_path + '/stopword/indonesian'
    )]

    @staticmethod
    def stemmer(contents):
        start = int(round(time.time() * 1000))
        print 'Start stemming from {} docs'.format(len(contents))

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        _stemmed = map(lambda content: stemmer.stem(content.lower()), contents)

        end = int(round(time.time() * 1000))
        print 'Done in : {} milliseconds'.format(end-start)

        return _stemmed

    @classmethod
    def remove_punctuation(cls, _contents):
        return map(lambda content: content.translate(None, cls.PUNCTUATION), _contents)

    @classmethod
    def remove_stopword(cls, _contents):
        start = int(round(time.time() * 1000))
        print 'Start stopwords removal from {} docs'.format(len(_contents))

        _contents_cleared = []
        for news in _contents:
            tokenize = news.split()
            _news = [word for word in tokenize if
                     word not in cls.STOPWORDS and
                     not word.startswith(string.digits)]
            _contents_cleared.append(' '.join(_news))

        end = int(round(time.time() * 1000))
        print 'Done in : {} milliseconds'.format(end-start)

        return _contents_cleared

    @classmethod
    def run(cls, contents):
        _contents = cls.stemmer(contents)
        _contents = cls.remove_punctuation(_contents)
        _contents = cls.remove_stopword(_contents)
        return _contents

if __name__ == '__main__':
    assert Preprocessor().run(
        contents=["Jakarta dibuat rusuh oleh pendemo",
                  "Macetnya Jogja menyerupai Jakarta"]
    ) == ['jakarta rusuh demo', 'macet jogja rupa jakarta'], 'We\'ve got problem dude'
