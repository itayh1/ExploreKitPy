
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
        self.indicesOfTrainingFolds = []
        self.indicesOfTestFolds = []

        self.maxNumOfDiscreteValuesForInstancesObject = maxNumOfValsPerDiscreteAttribtue
        # self.distinctValColumns = distinctValColumns

        for fold in folds:
            if not fold.isTestFold:
                self.indicesOfTrainingFolds.extend(fold.getIndices())

            else:
                self.indicesOfTestFolds.extend(fold.getIndices())


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

    # Used to obtain either the training or test set of the dataset
    def generateSet(self, getTrainingSet: bool):
        # ArrayList<Attribute> attributes = new ArrayList<>();

        # get all the attributes that need to be included in the set
        # Todo: takes only discrete and numeric columns
        # getAttributesListForClassifier(attributes)
        dfCopy = self.df.copy(deep=True)
        dfCopy.rename({})
        if getTrainingSet:
            finalSet = dfCopy.iloc[self.indicesOfTrainingFolds,:]
            finalSet.name = 'trainSet'
        else:
            finalSet = dfCopy.iloc[self.indicesOfTestFolds, :]
            finalSet.name = 'testSet'

        return finalSet
