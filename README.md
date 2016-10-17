# NBclassifier

Hello, I am learning in Naive Bayes Classifier. This is my project about text mining classifier in economy news. I have about one thousand news dataset. The data is csv file that consists 'media', 'date', 'url', and 'content'. I classified into 9 class of economy. 

I have problem with the accuracy of my classifier model. I used multinomial naive bayes classifier from scikitlearn.
There are my explanation about file/dataset :

1.File 'TFIDF-PREPROS(1).py' is file of preprocessing until TFIDF. In this file .py I used 'data_1332_tfidf.csv' and the result of tfidf is file 'tfidf1332.csv'
2.File 'kfold cross validation + klasifikasi.py' is file of my classifier model. 
   - Variabel y is values of 'klasifikasi' of my dataset : 'data_1332_9kelas.csv'
   - Variabel x is the result of TFIDF ('tfidf1332.csv')

My accuracy is stuck in 0.4-0.5 (40-50%) :(
