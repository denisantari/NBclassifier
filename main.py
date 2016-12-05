import time

import pandas as pd

from core.base import KFoldCrossValidation, RandomForestAlgorithm

if __name__ == '__main__':
    while True:
        print 'Hello wanderer'
        time.sleep(3)
        print 'Go sleep, you\'re tired'
        print 'We\'ll work when you sleep\nWhat do you want ?' \
              '\nPlease type kfold or get_news_classification'

        std_in = raw_input()
        if std_in == 'kfold':
            contents = pd.read_csv('data/stemming1332.csv')['hasil_stemming']
            labels = pd.read_csv('data/data_1332_9kelas.csv')['klasifikasi']

            KFoldCrossValidation().run(contents, labels)
        elif std_in == 'get_news_classification':
            print 'Please write down your news'
            content = raw_input()
            contents = content

            rfa = RandomForestAlgorithm()
            result = rfa.predict(contents)
            print 'This news belong to {} '.format(result)
            database_contents = (pd.read_csv('data/stemming1332.csv')['hasil_stemming']
                .tolist().append(contents))
            labels = pd.read_csv('data/data_1332_9kelas.csv')['klasifikasi'].\
                tolist().append(result)

            print 'Starting to update model...'
            rfa.update_model(database_contents, labels)
            print 'Done updating model...'
            break
