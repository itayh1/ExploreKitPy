import random

from sklearn.metrics import roc_auc_score

from Evaluation.Classifier import Classifier

import pandas as pd

from sklearn.model_selection import train_test_split

from Properties import Properties
from Utils.Loader import Loader


def main2():
    classifier = Classifier(Properties.classifier)
    df = pd.read_csv('temps.csv')
    size = df.shape[0]

    indicesOfTrainingFolds = random.sample(list(range(size)), int(size * 0.66))
    indicesOfTestFolds = list(set(list(range(size))) - set(indicesOfTrainingFolds))

    classifier.buildClassifier(df.iloc[indicesOfTrainingFolds,:])
    test_set = df.iloc[indicesOfTestFolds, :]
    evaluation_info = classifier.evaluateClassifier(test_set)
    score = roc_auc_score(test_set[df.columns[-1]],
                          evaluation_info.getScoreDistribution()[:, 1])
    print(score)

def main():
    baseFolder = '/home/itay/Documents/java/ExploreKit/AutomaticFeatureGeneration-master/ML_Background/Datasets/'
    german_credit_dataset_path = baseFolder + "german_credit.arff"
    loader = Loader()
    randomSeed = 42

    dataset = loader.readArff(german_credit_dataset_path, randomSeed, None, None, 0.66)
    df = dataset.df

    classifier = Classifier(Properties.classifier)
    classifier.buildClassifier(df.iloc[dataset.getIndicesOfTrainingInstances(), :])
    test_set = df.iloc[dataset.getIndicesOfTestInstances(), :]
    evaluation_info = classifier.evaluateClassifier(test_set)
    score = roc_auc_score(test_set[dataset.targetClass],
                          evaluation_info.getScoreDistribution()[:, 1])
    print(score)


if __name__ == '__main__':
    main()






