
from Classifier import Classifier

import pandas as pd
from pandas.api.types import CategoricalDtype

import numpy as np
from sklearn.model_selection import train_test_split

def main():
    features = pd.read_csv('temps.csv')
    # features = pd.get_dummies(features)
    features.rename({'actual': 'class'}, axis=1, inplace=True)

    print(features['class'])

    train, test = train_test_split(features, test_size=0.25, random_state=42)

    test = test[test['week']!='Sun']

    cls = Classifier('RandomForest')
    cls.buildClassifier(train)

    cls.evaluateClassifier(test)


if __name__ == '__main__':
    main()






