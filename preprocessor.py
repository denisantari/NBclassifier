import abc #library abstract base class, untuk asbtraksi kelas
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class Preprocessor(object):
    metaclass = abc.ABCMeta

    def __init__(self, contents): #inisiasi data, data yang dibuat pada self merupakan variabel obyek
        self._contents = contents
        
    """staticmethod digunakan untuk menggabungkan fungsi yg memupunyai beberapa logical connection with a class to the classs,
       disini maksudnya fungsi stemmer, punctuation dan stopword adalah fungsi yg saling berkaitan"""
    
    @staticmethod 
    def stemmer(_contents):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return map(lambda content: stemmer.stem(content.lower()), _contents)

    @staticmethod
    def punctuation_replace(_contents):
        return map(lambda content: content.translate(None, string.punctuation), _contents)

    @staticmethod
    def stopwords_remover(_contents):
        _contents_cleared = []
        for news in _contents:
            tokenize = news.split()
            _news = [word for word in tokenize if
                     word not in nltk.corpus.stopwords.words('indonesian') and
                     not word.startswith(string.digits)]
            _contents_cleared.append(' '.join(_news))

    def run(self):
        self.stemmer(self._contents)
        self.punctuation_replace(self._contents)
        self.stopwords_remover(self._contents)
        return self._contents


def tf_idf(_contents):
    vector = CountVectorizer(analyzer='word', min_df=0.01, max_df=.9, max_features=1000,
                             decode_error='replace')
    vector_train = vector.fit_transform(_contents)
    transformer = TfidfTransformer(norm='l2')
    return transformer.fit_transform(vector_train).toarray()

#
# print "DF", type(df['content']), "\n", df['content']
# isiberita = df['content'].tolist()
# print "DF list isiberita ", isiberita, type(isiberita)
# df.head()

# path = 'D:/SKRIPSI/percobaan/1332data9klas/data_1332_tfidf.csv'
#
# # now we get 300 words as vocab and content_final (content that has been cleared)
#
# # this can take some time, this is from sklearn tfidfVectorizer
# numpy.savetxt('D:/SKRIPSI/percobaan/tfidf1332.csv', tfidf_hasil.todense(), delimiter=',')
