from EvaluationInfo import EvaluationInfo
from Logger import Logger
from Properties import Properties

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pandas.api.types import CategoricalDtype

class Classifier:

    def __init__(self, classifier: str):
        if classifier == 'RandomForest':
            self.cls = RandomForestClassifier(random_state=Properties.randomSeed)
        else:
            msg = f'Unknown classifier: {classifier}'
            Logger.Error(msg)
            raise Exception(msg)

        self.categoricalColumnsMap: dict

    def buildClassifier(self, trainingSet: pd.DataFrame):
        X = trainingSet.drop(labels=['class'], axis=1)

        self._saveValuesOfCategoricalColumns(X)

        X = pd.get_dummies(X)

        y = trainingSet['class']
        self.cls.fit(X, y)

    def evaluateClassifier(self, testSet: pd.DataFrame) -> EvaluationInfo:
        X = testSet.drop(labels=['class'], axis=1)

        X = self._getDataframeWithCategoricalColumns(X)

        X = pd.get_dummies(X)

        # Returns ndarray of shape (n_samples, n_classes)
        scoresDist = self.cls.predict_proba(X)

        return EvaluationInfo(self.cls, scoresDist, testSet['class'])

    # Returns 2 lists, first is the the true/actual values and the second one is the predictions
    def predictClassifier(self, testSet: pd.DataFrame):
        X = testSet.drop(labels=['class'], axis=1)

        X = self._getDataframeWithCategoricalColumns(X)

        X = pd.get_dummies(X)

        # Returns ndarray of shape (n_samples, n_classes)
        preds = self.cls.predict(X)

        return testSet['class'].values, preds

    # Save categorical columns for one-hot encoding in test
    def _saveValuesOfCategoricalColumns(self, df: pd.DataFrame):
        self.categoricalColumnsMap = {col: df[col].unique() for col in df.select_dtypes(include='object').columns}

    # Set df's categorical columns to Categorical type to remember missing categories in test
    def _getDataframeWithCategoricalColumns(self, df: pd.DataFrame):
        for columnsName, categories in self.categoricalColumnsMap.items():
            df[columnsName] = df[columnsName].astype(CategoricalDtype(categories))
        return df
