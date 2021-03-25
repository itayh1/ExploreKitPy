
import numpy as np
import random

class Fold:

    indices = []
    numInstancesPerClass = []
    indicesByClass = []
    numOfInstancesInFold = 0
    distinctValMappings = {}
    typeOfFold = None

    def __init__(self, numOfClasses: int, isTestFold: bool):
        self.indices = []
        self.numInstancesPerClass = numOfClasses * [0]
        self.isTestFold = isTestFold

        self.indicesByClass = {i:[] for i in range(numOfClasses)}    # class -> indices
        self.distinctValMappings = None

    def getIndices(self):
        return self.indices

    def setIsTestFold(self, isTest: bool):
        self.isTestFold = isTest

    '''
    Used to gnerate a sub-fold by randomly sampling a predefined number of samples from the fold. In the case
    of distinct values, the the number of samples referred to the distinct values and all the associated indices
    will be added.
    '''
    def generateSubFold(self, numOfSamples, randomSeed):
        subFold = Fold(self.numInstancesPerClass.length, self.typeOfFold)

        # determine how many instances of each class needs to be added
        requiredNumOfSamplesPerClass = self.getRequiredNumberOfInstancesPerClass(numOfSamples)

        # now we need to randomly select the samples
        random.seed(randomSeed)
        for i in range(0, len(self.numInstancesPerClass), 1):
            if (len(self.distinctValMappings) == 0):
                selectedIndicesPerClass = []
                while (len(selectedIndicesPerClass) < requiredNumOfSamplesPerClass[i]):
                    instanceIndex = self.indicesByClass[i].get(random.randint(0, len(self.indicesByClass[i])))
                    if (not instanceIndex in selectedIndicesPerClass):
                        selectedIndicesPerClass.append(instanceIndex)
                        subFold.addInstance(instanceIndex, i)

            else:
                keySetValues = self.init_list(len(self.distinctValMappings.keys()), "")
                counter = 0
                for key in self.distinctValMappings.keys():
                    keySetValues[counter] = key
                    counter += 1

                selectedIndicesPerClass = []
                while (len(selectedIndicesPerClass) < requiredNumOfSamplesPerClass[i]):
                    distictValKey = keySetValues[random.randint(0, len(keySetValues))]
                    if (not distictValKey in selectedIndicesPerClass and
                            self.distinctValMappings.get(distictValKey)[0] in self.indicesByClass[i]):
                        selectedIndicesPerClass.append(distictValKey)
                        subFold.addDistinctValuesBatch(distictValKey, self.distinctValMappings.get(distictValKey),
                                                       i)

        return subFold

    def getRequiredNumberOfInstancesPerClass(self, numOfSamples):
        numOfInstancesPerClass = self.init_list(len(self.numInstancesPerClass), 0.0)

        # If there are no distinct values, the problem is simple
        if (len(self.distinctValMappings) == 0):
            for i in range(0, len(numOfInstancesPerClass), 1):
                numOfInstancesPerClass[i] = (float(self.numInstancesPerClass[i]) / float(
                    sum(self.numInstancesPerClass))) * numOfSamples

        else:
            # We need to find the number of DISTINCT VALUES per class
            for item in self.distinctValMappings.keys():
                index = self.distinctValMappings.get(item)[0]
                for i in range(0, len(self.indicesByClass), 1):
                    if index in self.indicesByClass[i]:
                        numOfInstancesPerClass[i] += 1
                        break
            for i in range(0, len(numOfInstancesPerClass), 1):
                numOfInstancesPerClass[i] = (numOfInstancesPerClass[i] / sum(numOfInstancesPerClass)) * numOfSamples

        return numOfInstancesPerClass
