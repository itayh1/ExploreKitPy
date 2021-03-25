
import numpy as np
import pandas as pd

from Dataset import Dataset
from FilterWrapperHeuristicSearch import FilterWrapperHeuristicSearch
from Logger import Logger




class DatasetBasedAttributes:

    # Basic information on the dataset
    numOfInstances: float
    numOfClasses: float
    numOfFeatures: float
    numOfNumericAtributes: float
    numOfDiscreteAttributes: float
    ratioOfNumericAttributes: float
    ratioOfDiscreteAttributes: float
    minorityClassPercentage: float

    # discrete features-specific attributes (must not include the target class)
    maxNumberOfDiscreteValuesPerAttribute: float
    minNumberOfDiscreteValuesPerAttribtue: float
    avgNumOfDiscreteValuesPerAttribute: float
    stdevNumOfDiscreteValuesPerAttribute: float

    # Statistics on the initial performance of the dataset
    numOfFoldsInEvaluation: float
    maxAUC: float
    minAUC: float
    avgAUC: float
    stdevAUC: float

    maxLogLoss: float
    minLogLoss: float
    avgLogLoss: float
    stdevLogLoss: float

    maxPrecisionAtFixedRecallValues: dict
    minPrecisionAtFixedRecallValues: dict
    avgPrecisionAtFixedRecallValues: dict
    stdevPrecisionAtFixedRecallValues: dict


    # Statistics on the initial attributes' entropy with regards to the target class and their interactions
    maxIGVal: float
    minIGVal: float
    avgIGVal: float
    stdevIGVal: float

    discreteAttsMaxIGVal: float
    discreteAttsMinIGVal: float
    discreteAttsAvgIGVal: float
    discreteAttsStdevIGVal: float

    numericAttsMaxIGVal: float
    numericAttsMinIGVal: float
    numericAttsAvgIGVal: float
    numericAttsStdevIGVal: float

    # Statistics on the correlation of the dataset's features
    maxPairedTTestValueForNumericAttributes: float
    minPairedTTestValueForNumericAttributes: float
    avgPairedTTestValueForNumericAttributes: float
    stdevPairedTTestValueForNumericAttributes: float

    maxChiSquareValueforDiscreteAttributes: float
    minChiSquareValueforDiscreteAttributes: float
    avgChiSquareValueforDiscreteAttributes: float
    stdevChiSquareValueforDiscreteAttributes: float

    maxChiSquareValueforDiscreteAndDiscretizedAttributes: float
    minChiSquareValueforDiscreteAndDiscretizedAttributes: float
    avgChiSquareValueforDiscreteAndDiscretizedAttributes: float
    stdevChiSquareValueforDiscreteAndDiscretizedAttributes: float

    # support parameters - not to be included in the output of the class
    discreteAttributesList: list
    numericAttributesList: list

    def getDatasetBasedFeatures(self, dataset: Dataset, classifier: str) -> dict:
        try:
            self.processGeneralDatasetInfo(dataset)

            self.processInitialEvaluationInformation(dataset, classifier);

            processEntropyBasedMeasures(dataset, properties);

            processAttributesStatisticalTests(dataset);

            return generateDatasetAttributesMap();

        except Exception as ex:
            Logger.Error(f'Failed in func "getDatasetBasedFeatures" with exception: {ex}')

        return None

    def processGeneralDatasetInfo(self, dataset: Dataset):
        self.numOfInstances = dataset.getNumOfInstancesPerColumn()

        # If an index to the target class was not provided, it's the last attirbute.
        self.numOfClasses = dataset.getNumOfClasses()
        self.numOfFeatures = dataset.getAllColumns(False).size() # the target class is not included
        self.numOfNumericAtributes = 0
        self.numOfDiscreteAttributes = 0

        for columnInfo in dataset.getAllColumns(False):
            if pd.api.types.is_float_dtype(columnInfo):
                self.numOfNumericAtributes += 1
                self.numericAttributesList.append(columnInfo)

            if pd.api.types.is_integer_dtype(columnInfo):
                self.numOfDiscreteAttributes += 1
                self.discreteAttributesList.append(columnInfo)


        self.ratioOfNumericAttributes = self.numOfNumericAtributes / (self.numOfNumericAtributes + self.numOfDiscreteAttributes)
        self.ratioOfDiscreteAttributes = self.ratioOfDiscreteAttributes / (self.numOfNumericAtributes + self.numOfDiscreteAttributes)

        # TODO check minority
        numOfAllClassItems = dataset.getNumOfTrainingDatasetRows()
        numOfMinorityClassItems = dataset.getNumOfRowsPerClassInTrainingSet()[dataset.getMinorityClassIndex()]

        self.minorityClassPercentage = (float)((numOfMinorityClassItems / numOfAllClassItems) * 100)

        numOfValuesperDiscreteAttribute = []
        for columnInfo in self.discreteAttributesList:
            numOfValuesperDiscreteAttribute.append(columnInfo.values.unique.size)
            # numOfValuesperDiscreteAttribute.append((float)((DiscreteColumn)columnInfo.getColumn()).getNumOfPossibleValues())

        if len(numOfValuesperDiscreteAttribute) > 0:
            self.maxNumberOfDiscreteValuesPerAttribute = max(numOfValuesperDiscreteAttribute)
            self.minNumberOfDiscreteValuesPerAttribtue = min(numOfValuesperDiscreteAttribute)
            self.avgNumOfDiscreteValuesPerAttribute = sum(numOfValuesperDiscreteAttribute)/len(numOfValuesperDiscreteAttribute)
            # the stdev requires an interim step
            self.stdevNumOfDiscreteValuesPerAttribute = np.asarray(numOfValuesperDiscreteAttribute, dtype=np.float32).std()
            # tempStdev = numOfValuesperDiscreteAttribute.stream().mapToDouble(a -> Math.pow(a - avgNumOfDiscreteValuesPerAttribute, 2)).sum()
            # self.stdevNumOfDiscreteValuesPerAttribute = Math.sqrt(tempStdev / numOfValuesperDiscreteAttribute.size())

        else:
            self.maxNumberOfDiscreteValuesPerAttribute = 0
            self.minNumberOfDiscreteValuesPerAttribtue = 0
            self.avgNumOfDiscreteValuesPerAttribute = 0
            self.stdevNumOfDiscreteValuesPerAttribute = 0

    def processInitialEvaluationInformation(self, dataset: Dataset, classifier: str):
        # We now need to test all folds combinations (the original train/test allocation is disregarded, which is
        # not a problem for the offline training. The test set dataset MUST submit a new dataset object containing
        # only the training folds
        for fold in dataset.getFolds():
            fold.setIsTestFold(False)

        fwhs = FilterWrapperHeuristicSearch(10)
        wrapperEvaluator = fwhs.getWrapper("AucWrapperEvaluator")
        leaveOneFoldOutDatasets = dataset.GenerateTrainingSetSubFolds()
        classificationResults = wrapperEvaluator.produceClassificationResults(leaveOneFoldOutDatasets)

        aucVals = []
        logLossVals = []
        recallPrecisionValues = [] # list of dicts
        for classificationResult in classificationResults:
            aucVals.append(classificationResult.getAuc())
            logLossVals.append(classificationResult.getLogLoss())
            recallPrecisionValues.append(classificationResult.getRecallPrecisionValues())

        self.numOfFoldsInEvaluation = dataset.getFolds().size()

        aucVals = np.asarray(aucVals, dtype=np.float32)
        self.maxAUC = aucVals.max()
        self.minAUC = aucVals.min()
        self.avgAUC = np.average(aucVals)
        self.stdevAUC = aucVals.std()
        # double tempStdev = aucVals.stream().mapToDouble(a -> Math.pow(a - self.avgAUC, 2)).sum();
        # self.stdevAUC = Math.sqrt(tempStdev / aucVals.size());

        logLossVals = np.asarray(logLossVals, dtype=np.float32)
        self.maxLogLoss = logLossVals.max()
        self.minLogLoss = logLossVals.min()
        self.avgLogLoss = np.average(logLossVals)
        self.stdevLogLoss = logLossVals.std()
        # tempStdev = logLossVals.stream().mapToDouble(a -> Math.pow(a - self.avgLogLoss, 2)).sum();
        # self.stdevLogLoss = Math.sqrt(tempStdev / logLossVals.size());

        for recallVal in recallPrecisionValues[0].keys():
            maxVal = -1
            minVal = 2;
            valuesList = []
            for precisionRecallVals in recallPrecisionValues:
                maxVal = max(precisionRecallVals.get(recallVal), maxVal)
                minVal = min(precisionRecallVals.get(recallVal), minVal)
                valuesList.append(precisionRecallVals[recallVal])

            # now the assignments
            self.maxPrecisionAtFixedRecallValues[recallVal] = maxVal
            self.minPrecisionAtFixedRecallValues[recallVal] = minVal
            self.avgPrecisionAtFixedRecallValues.put(recallVal, valuesList.stream().mapToDouble(a -> a).average().getAsDouble());
            tempStdev = valuesList.stream().mapToDouble(a -> Math.pow(a - avgPrecisionAtFixedRecallValues.get(recallVal), 2)).sum();
            stdevPrecisionAtFixedRecallValues.put(recallVal, Math.sqrt(tempStdev / valuesList.size()));


