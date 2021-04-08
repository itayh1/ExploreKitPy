from random import Random

from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from Logger import Logger
from Dataset import Dataset
from Properties import Properties
from Fold import Fold


class Loader:

    def readArffAsDataframe(self, filePath: str):
        data = arff.loadarff(filePath)
        df = pd.DataFrame(data[0])
        return df

    def readArff(self, filePath: str, randomSeed: int, distinctValIndices: list, classAttIndex: str, trainingSetPercentageOfDataset: float) -> Dataset:

        try:
            data = arff.loadarff(filePath)
            df = pd.DataFrame(data[0])

            Logger.Info(f'num of attributes: {len(df.keys())}')
            Logger.Info(f'num of instances: {len(df.values)}')


            if (classAttIndex == None) or (classAttIndex == ''):
                targetClassName = df.keys()[-1]
            else:
                targetClassName = classAttIndex
            df[targetClassName] = df[targetClassName].astype(np.int_) - 1

            if distinctValIndices == None:
                folds = self.GenerateFolds(df[targetClassName], randomSeed, trainingSetPercentageOfDataset)
            else:
                pass    #TODO: missing func?

            distinctValColumnInfos = []
            if distinctValIndices != None:
                for distinctColumnIndex in distinctValIndices:
                    distinctValColumnInfos.append(df[distinctColumnIndex])

            # Fially, we can create the Dataset object
            return Dataset(df, folds, targetClassName, data[1].name, randomSeed, Properties.maxNumberOfDiscreteValuesForInclusionInSet)

        except Exception as ex:
            Logger.Error(f'Exception in readArff. message: {ex}')
            return None

    def GenerateFolds(self, targetColumnInfo, randomSeed: int, trainingSetPercentage: float) -> list: #List<Fold>

        # Next, we need to get the number of classes (we assume the target class is discrete)
        numOfClasses = targetColumnInfo.unique().shape[0]

        # Store the indices of the instances, partitioned by their class
        itemIndicesByClass = [[] for i in range(numOfClasses)]


        for i  in range(targetColumnInfo.shape[0]):
            instanceClass = int(targetColumnInfo[i])
            itemIndicesByClass[instanceClass].append(i)

        # Now we calculate the number of instances from each class we want to assign to fold
        numOfFolds = Properties.numOfFolds
        maxNumOfInstancesPerTrainingClassPerFold = [] # np.arr numOfClasses];
        maxNumOfInstancesPerTestClassPerFold = [] # new double[numOfClasses];
        for i in range(len(itemIndicesByClass)):
            # If the training set overall size (in percentages) is predefined, use it. Otherwise, just create equal folds
            if trainingSetPercentage == -1:
                maxNumOfInstancesPerTrainingClassPerFold[i] = len(itemIndicesByClass[i])/numOfFolds
                maxNumOfInstancesPerTestClassPerFold[i] = len(itemIndicesByClass[i])/numOfFolds

            else:
                # The total number of instances, multipllied by the training percentage and then divided by the number of the TRAINING folds
                maxNumOfInstancesPerTrainingClassPerFold.append(len(itemIndicesByClass[i]) * trainingSetPercentage /(numOfFolds-1))
                maxNumOfInstancesPerTestClassPerFold.append(len(itemIndicesByClass[i]) - maxNumOfInstancesPerTrainingClassPerFold[i])

        # We're using a fixed seed so we can reproduce our results
        # int randomSeed = Integer.parseInt(properties.getProperty("randomSeed"))
        rnd = Random(randomSeed)

        # Now create the Fold objects and start filling them
        folds = [] #new ArrayList<>(numOfClasses);
        for i in range(numOfFolds):
            isTestFold = self.designateFoldAsTestSet(numOfFolds, i, Properties.testFoldDesignation)
            fold = Fold(numOfClasses, isTestFold)
            folds.append(fold)


        for i in range(targetColumnInfo.shape[0]):
            instanceClass = targetColumnInfo[i]

            foundAssignment = False
            exploredIndices = []
            while not foundAssignment:
                # We now randomly sample a fold and see whether the instance can be assigned to it. If not, sample again
                foldIdx = rnd.randrange(numOfFolds)
                if str(foldIdx) not in exploredIndices:
                    exploredIndices.append(str(foldIdx))

                # Now see if the instance can be assigned to the fold
                fold = folds[foldIdx]
                if not fold.isTestFold:
                    if fold.getNumOfInstancesPerClass(instanceClass) < maxNumOfInstancesPerTrainingClassPerFold[instanceClass] or len(exploredIndices) == numOfFolds:
                        fold.addInstance(i, instanceClass)
                        foundAssignment = True

                else:
                    if fold.getNumOfInstancesPerClass(instanceClass) < maxNumOfInstancesPerTestClassPerFold[instanceClass] or len(exploredIndices) == numOfFolds:
                        fold.addInstance(i, instanceClass)
                        foundAssignment = True

        return folds

    def getFolds(df: pd.DataFrame, targetClassname: str, k: int) -> list:
        numOfClasses = df[targetClassname].nunique()
        return [Fold(numOfClasses, False) for i in range(k)]

    def designateFoldAsTestSet(self, numOfFolds: int, currentFoldIdx: int, designationMethod: str):
         if designationMethod == 'last':
             if currentFoldIdx == (numOfFolds - 1):
                return True
             else:
                return False
         else:
             raise Exception("unknown test fold selection method")
