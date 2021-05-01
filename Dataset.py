
import numpy as np
import pandas as pd
import Fold

class Dataset:

    def __init__(self, df: pd.DataFrame, folds: list, targetClass: str, name: str, seed: int, maxNumOfValsPerDiscreteAttribtue: int):
        self.randomSeed = seed
        self.df = df
        self.folds = folds
        self.targetClass = targetClass
        self.name = name
        self.indicesOfTrainingFolds = None
        self.indicesOfTestFolds = None

        self.maxNumOfDiscreteValuesForInstancesObject = maxNumOfValsPerDiscreteAttribtue
        # self.distinctValColumns = distinctValColumns


    def getNumOfInstancesPerColumn(self):
        pass

    def getNumOfClasses(self):
        pass

    def getAllColumns(self, includeTargetColumn: bool):
        pass

    def getNumOfTrainingDatasetRows(self):
        pass

    def getNumOfRowsPerClassInTrainingSet(self):
        pass

    def getMinorityClassIndex(self):
        pass

    def getFolds(self):
        pass

    def GenerateTrainingSetSubFolds(self):
        pass

    def getIndicesOfTrainingInstances(self):
        pass

    def getNumberOfRows(self):
        pass

    def getIndices(self):
        pass

    # Returns the columns used to create the distinct value of the instances
    # def getDistinctValueColumns(self):
    #     return self.distinctValColumns

    def getDistinctValueCompliantColumns(self):
        pass
