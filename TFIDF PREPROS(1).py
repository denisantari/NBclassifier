
# coding: utf-8

# In[1]:

import string
import os
import pandas as pd
from pandas import DataFrame, read_csv

data = r'D:/SKRIPSI/percobaan/1332data9klas/data_1332_tfidf.csv'
df = pd.read_csv(data)

print "DF", type (df['content']), "\n", df['content']
isiberita = df['content'].tolist()
print "DF list isiberita ", isiberita, type(isiberita)
df.head()


# In[2]:

import nltk
import string
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from collections import Counter

path = 'D:/SKRIPSI/percobaan/1332data9klas/data_1332_tfidf.csv'
token_dict = {}

factory = StemmerFactory()
stemmer = factory.create_stemmer()

content_stemmed = map(lambda x: stemmer.stem(x), isiberita)
content_no_punc = map(lambda x: x.lower().translate(None, string.punctuation), content_stemmed)
content_final = [] # add final list variable for content after stop words removal and digit removal

# looping every news to perform stop words removal
for news in content_no_punc: 
	word_token = nltk.word_tokenize(news) # get word token for every news (split news into each separate words)
	word_token = [word for word in word_token if not word in nltk.corpus.stopwords.words('indonesian') and not word[0].isdigit()] # remove indonesian stop words and number
	content_final.append(" ".join(word_token)) # each news has been separated like ["saham", "indonesia"] so we can join them with "space", so it will be ["saham indonesia"] 

# now we get content cleared from all stop words, string punctuation, and digit. So we should find a list of vocab that we can use to avoid over-features while count tfidf

counter = Counter() # counter initiate
[counter.update(news.split()) for news in content_final] # we split every news to get counter of each words
print(counter.most_common(300)) # we look 300 potential words to be a vocab ("indonesia", 20), ("jakarta", 19), etc
vocab = counter.most_common(300)
vocab = [word[:][0] for word in vocab] # we get vocab in each word in 100 most_common features, [:] is for all row, [0] is for index 0

print(vocab)

# now we get 300 words as vocab and content_final (content that has been cleared)

#this can take some time, this is from sklearn tfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word', stop_words=nltk.corpus.stopwords.words('indonesian'), ngram_range=(1,1), min_df=0.04, vocabulary=vocab)
tfidf_hasil = tfidf.fit_transform(content_final)
features = tfidf.get_feature_name()
print(features)
print(tfidf_hasil.toarray())



# In[5]:

import numpy
numpy.savetxt('D:/SKRIPSI/percobaan/tfidf1332.csv', tfidf_hasil.todense(), delimiter=',')


# In[1]:

#df = pd.DataFrame(data = vocab)
#df


# In[14]:

#df.to_csv('D:/SKRIPSI/fitur300.csv')


# In[ ]:



