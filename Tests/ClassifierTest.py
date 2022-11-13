
from Evaluation.Classifier import Classifier

import pandas as pd

from sklearn.model_selection import train_test_split

def main():
    features = pd.read_csv('temps.csv')
    features.rename({'actual': 'class'}, axis=1, inplace=True)

    # print(features['class'])

    train, test = train_test_split(features, test_size=0.25, random_state=42)

    # test = test[test['week']!='Sun']

    cls = Classifier('RandomForest')


    # train
    cls.buildClassifier(train)

    preds = cls.predictClassifier(test)
    # test = test.drop(labels=['class'], axis=1)
    # res = cls.cls.predict(test)
    print(preds)
    # # evaluate
    # cls.evaluateClassifier(test)


if __name__ == '__main__':
    main()






