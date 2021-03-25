from scipy.io import arff
import pandas as pd
from sklearn.model_selection import KFold

from Logger import Logger
from Dataset import Dataset
from Properties import Properties
from Fold import Fold

class Loader:
    def getFolds(df: pd.DataFrame, targetClassname: str, k: int) -> list:
        numOfClasses = df[targetClassname].nunique()
        return [Fold(numOfClasses, False) for i in range(k)]

    def readArffAsDataframe(self, filePath: str):
        data = arff.loadarff(filePath)
        df = pd.DataFrame(data[0])
        return df

    def readArff(self, filePath: str, randomSeed: int, distinctValIndices: int, classAttIndex: str, trainingSetPercentageOfDataset: float) -> Dataset:

        try:
            data = arff.loadarff(filePath)
            df = pd.DataFrame(data[0])

            Logger.Info(f'num of attributes: {len(df.keys())}')
            Logger.Info(f'num of instances: {len(df.values)}')

            targetClassName = classAttIndex
            if (classAttIndex == None) or (classAttIndex == ''):
                targetClassName = df.keys()[-1]

            folds = Loader.getFolds(df, targetClassName, 4)
            return Dataset(df, folds, targetClassName, data[1].name, randomSeed, Properties.maxNumberOfDiscreteValuesForInclusionInSet)


        except Exception as ex:
            Logger.Error(f'Exception in readArff. message: {ex}')
            return None
