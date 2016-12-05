import abc  #library abstract base class, untuk asbtraksi kelas
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

    punctuation = string.punctuation
    stopwords = [line.rstrip('\n') for line in open(
        absolute_path + '/stopword/indonesian'
    )]
    digits = string.digits

    @staticmethod
    def stemmer(content):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stemmed = stemmer.stem(content)

        return stemmed

    @classmethod
    def remove_punctuation(cls, content):
        return content.translate(None, cls.punctuation)

    @classmethod
    def remove_stopword(cls, content):
        tokenize = content.split()
        content_cleared = [word for word in tokenize if word not in cls.stopwords and
                           not word.startswith(cls.digits)]

        return ' '.join(content_cleared)

    @classmethod
    def run(cls, contents):

        start = int(round(time.time()))
        print 'Start preprocessing from {} docs'.format(len(contents))

        _contents = []
        contents = (
            [contents] if not isinstance(contents, list) else (contents)
        )
        for content in contents:
            content = content.lower()
            content = cls.stemmer(content)
            content = cls.remove_punctuation(content)
            content = cls.remove_stopword(content)
            _contents.append(content)

        end = int(round(time.time()))
        print 'Done in : {} seconds'.format(end-start)

        return _contents

if __name__ == '__main__':
    assert Preprocessor().run(
        contents=["Jakarta dibuat rusuh oleh pendemo",
                  "Macetnya Jogja menyerupai Jakarta"]
    ) == ['jakarta rusuh demo', 'macet jogja rupa jakarta'], 'We\'ve got problem dude'
