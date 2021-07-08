
import numpy as np
import pandas as pd
from Fold import Fold

class Dataset:

    def __init__(self, df: pd.DataFrame, folds: list, targetClass: str, name: str, seed: int, maxNumOfValsPerDiscreteAttribtue: int):
        self.randomSeed = seed
        self.df = df
        self.folds = folds
        self.targetClass = targetClass
        self.name = name
        self.indicesOfTrainingFolds = []
        self.indicesOfTestFolds = []
        self.classes = df[targetClass].unique()
        self.numOfTrainingInstancesPerClass = dict.fromkeys(self.classes, 0)
        self.numOfTestInstancesPerClass = dict.fromkeys(self.classes, 0)

        self.maxNumOfDiscreteValuesForInstancesObject = maxNumOfValsPerDiscreteAttribtue
        # self.distinctValColumns = distinctValColumns

        for fold in folds:
            if not fold.isTestFold:
                self.indicesOfTrainingFolds.extend(fold.getIndices())
                for cls in self.classes:
                    self.numOfTrainingInstancesPerClass[cls] += fold.getNumOfInstancesPerClass(cls)
            else:
                self.indicesOfTestFolds.extend(fold.getIndices())
                for cls in self.classes:
                    self.numOfTestInstancesPerClass[cls] += fold.getNumOfInstancesPerClass(cls)


    def getNumOfInstancesPerColumn(self):
        return self.df.shape[0]

    def getNumOfClasses(self):
        return self.df[self.targetClass].unique().shape[0]

    # Return number of features not including target column
    def getNumOfFeatures(self):
        return self.df.shape[1] - 1

    def getAllColumns(self, includeTargetColumn: bool) -> pd.DataFrame:
        if includeTargetColumn:
            columns = self.df.columns
        else:
            columns = self.df.columns.drop(self.targetClass)
        return self.df[columns]

    # return list of types('columnName', dtype)
    def getColumnsDtypes(self, includeTargetColumn: bool):
        if includeTargetColumn:
            columns = self.df.dtypes
        else:
            columns = self.df.dtypes.drop(self.targetClass)
        return columns.items()

    def getNumOfTrainingDatasetRows(self):
        return len(self.indicesOfTrainingFolds)

    def getNumOfRowsPerClassInTrainingSet(self) -> dict:
        return self.numOfTrainingInstancesPerClass

    def getNumOfRowsPerClassInTestSet(self) -> dict:
        return self.numOfTestInstancesPerClass

    def getMinorityClassIndex(self) -> str:
        return min(self.numOfTrainingInstancesPerClass.keys(),
                   key=(lambda key: self.numOfTrainingInstancesPerClass[key]))

    def getFolds(self):
        return self.folds

    def getTargetClassColumn(self):
        return self.df[self.targetClass]

    # Partitions the training folds into a set of LOO folds. One of the training folds is designated as "test",
    # while the remaining folds are used for training. All possible combinations are returned.
    # return list of Dataset
    def GenerateTrainingSetSubFolds(self):
        trainingFolds = []
        for fold in self.folds:
            if not fold.isTestFold:
                trainingFolds.append(fold)

        trainingDatasets = []
        for i in range(len(trainingFolds)):
            newFoldsList = []
            for j in range(len(trainingFolds)):
                currentFold = trainingFolds[j]
                # if i==j, then this is the test fold
                newFold = Fold(self.classes,(i==j))
                newFold.setIndices(currentFold.getIndices())
                # newFold.setNumOfInstancesInFold(currentFold.getNumOfInstancesInFold())
                newFold.setInstancesClassDistribution(currentFold.getInstancesClassDistribution())
                newFold.setIndicesPerClass(currentFold.getIndicesPerClass())
                # newFold.setDistinctValMappings(currentFold.getDistinctValMappings())
                newFoldsList.append(newFold)

            # now that we have the folds, we can generate the Dataset object
            subDataset = Dataset(self.df, newFoldsList, self.targetClass, self.name,self.randomSeed, self.maxNumOfDiscreteValuesForInstancesObject)
            trainingDatasets.append(subDataset)
        return trainingDatasets

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
    def generateSet(self, getTrainingSet: bool) -> pd.DataFrame:
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

    def replicateDatasetByColumnIndices(self, indices: list):
        newDataset = Dataset(self.df.copy(), self.folds, self.targetClass, self.name,
                             self.randomSeed, self.maxNumOfDiscreteValuesForInstancesObject)
        return newDataset
