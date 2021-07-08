
import numpy as np
import pandas as pd
import scipy
import scipy.stats

from AttributeInfo import AttributeInfo
from AucWrapperEvaluator import AucWrapperEvaluator
from Column import Column
from Dataset import Dataset
from Operators.EqualRangeDiscretizerUnaryOperator import EqualRangeDiscretizerUnaryOperator
# from FilterWrapperHeuristicSearch import FilterWrapperHeuristicSearch
from InformationGainFilterEvaluator import InformationGainFilterEvaluator
from Logger import Logger
from Properties import Properties


class DatasetBasedAttributes:

    def __init__(self):
        # Basic information on the dataset
        self.numOfInstances: float
        self.numOfClasses: float
        self.numOfFeatures: float
        self.numOfNumericAtributes: float
        self.numOfDiscreteAttributes: float
        self.ratioOfNumericAttributes: float
        self.ratioOfDiscreteAttributes: float
        self.minorityClassPercentage: float

        # discrete features-specific attributes (must not include the target class)
        self.maxNumberOfDiscreteValuesPerAttribute: float
        self.minNumberOfDiscreteValuesPerAttribtue: float
        self.avgNumOfDiscreteValuesPerAttribute: float
        self.stdevNumOfDiscreteValuesPerAttribute: float

        # Statistics on the initial performance of the dataset
        self.numOfFoldsInEvaluation: float
        self.maxAUC: float
        self.minAUC: float
        self.avgAUC: float
        self.stdevAUC: float

        self.maxLogLoss: float
        self.minLogLoss: float
        self.avgLogLoss: float
        self.stdevLogLoss: float

        self.maxPrecisionAtFixedRecallValues: dict
        self.minPrecisionAtFixedRecallValues: dict
        self.avgPrecisionAtFixedRecallValues: dict
        self.stdevPrecisionAtFixedRecallValues: dict


        # Statistics on the initial attributes' entropy with regards to the target class and their interactions
        self.maxIGVal: float
        self.minIGVal: float
        self.avgIGVal: float
        self.stdevIGVal: float

        self.discreteAttsMaxIGVal: float
        self.discreteAttsMinIGVal: float
        self.discreteAttsAvgIGVal: float
        self.discreteAttsStdevIGVal: float

        self.numericAttsMaxIGVal: float
        self.numericAttsMinIGVal: float
        self.numericAttsAvgIGVal: float
        self.numericAttsStdevIGVal: float

        # Statistics on the correlation of the dataset's features
        self.maxPairedTTestValueForNumericAttributes: float
        self.minPairedTTestValueForNumericAttributes: float
        self.avgPairedTTestValueForNumericAttributes: float
        self.stdevPairedTTestValueForNumericAttributes: float

        self.maxChiSquareValueforDiscreteAttributes: float
        self.minChiSquareValueforDiscreteAttributes: float
        self.avgChiSquareValueforDiscreteAttributes: float
        self.stdevChiSquareValueforDiscreteAttributes: float

        self.maxChiSquareValueforDiscreteAndDiscretizedAttributes: float
        self.minChiSquareValueforDiscreteAndDiscretizedAttributes: float
        self.avgChiSquareValueforDiscreteAndDiscretizedAttributes: float
        self.stdevChiSquareValueforDiscreteAndDiscretizedAttributes: float

        # support parameters - not to be included in the output of the class
        self.discreteAttributesList: list
        self.numericAttributesList: list

    def getDatasetBasedFeatures(self, dataset: Dataset, classifier: str) -> dict:
        try:
            self.processGeneralDatasetInfo(dataset)

            self.processInitialEvaluationInformation(dataset, classifier)

            self.processEntropyBasedMeasures(dataset)

            self.processAttributesStatisticalTests(dataset)

            return self.generateDatasetAttributesMap()

        except Exception as ex:
            Logger.Error(f'Failed in func "getDatasetBasedFeatures" with exception: {ex}')

        return None

    # Extracts general information regarding the analyzed dataset
    def processGeneralDatasetInfo(self, dataset: Dataset):
        self.numOfInstances = dataset.getNumOfInstancesPerColumn()

        # If an index to the target class was not provided, it's the last attirbute.
        self.numOfClasses = dataset.getNumOfClasses()
        self.numOfFeatures = dataset.getNumOfFeatures() # dataset.getAllColumns(False).size() # the target class is not included
        self.numOfNumericAtributes = 0
        self.numOfDiscreteAttributes = 0
        self.numericAttributesList = []
        self.discreteAttributesList = []

        for columnName, columnType in dataset.getColumnsDtypes(False):
            if pd.api.types.is_float_dtype(columnType):
                self.numOfNumericAtributes += 1
                self.numericAttributesList.append(columnName)

            elif pd.api.types.is_integer_dtype(columnType):
                self.numOfDiscreteAttributes += 1
                self.discreteAttributesList.append(columnName)


        self.ratioOfNumericAttributes = self.numOfNumericAtributes / (self.numOfNumericAtributes + self.numOfDiscreteAttributes)
        self.ratioOfDiscreteAttributes = self.numOfDiscreteAttributes / (self.numOfNumericAtributes + self.numOfDiscreteAttributes)

        # TODO check minority
        numOfAllClassItems = dataset.getNumOfTrainingDatasetRows()
        numOfMinorityClassItems = dataset.getNumOfRowsPerClassInTrainingSet()[dataset.getMinorityClassIndex()]

        self.minorityClassPercentage = ((numOfMinorityClassItems / numOfAllClassItems) * 100)

        numOfValuesperDiscreteAttribute = []
        for columnInfo in self.discreteAttributesList:
            numOfValuesperDiscreteAttribute.append(dataset.df[columnInfo].unique().shape[0])
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

    # Used to obtain information about the performance of the classifier on the initial dataset. For training
    # datasets the entire dataset needs to be provided. For test datasets - only the training folds.
    def processInitialEvaluationInformation(self, dataset: Dataset, classifier: str):
        # We now need to test all folds combinations (the original train/test allocation is disregarded, which is
        # not a problem for the offline training. The test set dataset MUST submit a new dataset object containing
        # only the training folds
        for fold in dataset.getFolds():
            fold.setIsTestFold(False)

        wrapperName = 'AucWrapperEvaluator'
        if wrapperName == 'AucWrapperEvaluator':
            wrapperEvaluator = AucWrapperEvaluator()
        else:
            raise Exception('Unidentified wrapper')

        leaveOneFoldOutDatasets = dataset.GenerateTrainingSetSubFolds()
        classificationResults = wrapperEvaluator.produceClassificationResults(leaveOneFoldOutDatasets)

        aucVals = []
        logLossVals = []
        recallPrecisionValues = [] # list of dicts
        for classificationResult in classificationResults:
            aucVals.append(classificationResult.getAuc())
            logLossVals.append(classificationResult.getLogLoss())
            recallPrecisionValues.append(classificationResult.getRecallPrecisionValues())

        self.numOfFoldsInEvaluation = len(dataset.getFolds())

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

        self.maxPrecisionAtFixedRecallValues = {}
        self.minPrecisionAtFixedRecallValues = {}
        self.avgPrecisionAtFixedRecallValues = {}
        self.stdevPrecisionAtFixedRecallValues = {}

        for recallVal in recallPrecisionValues[0].keys():
            maxVal = -1
            minVal = 2
            valuesList = []
            for precisionRecallVals in recallPrecisionValues:
                maxVal = max(precisionRecallVals.get(recallVal), maxVal)
                minVal = min(precisionRecallVals.get(recallVal), minVal)
                valuesList.append(precisionRecallVals[recallVal])

            # now the assignments
            self.maxPrecisionAtFixedRecallValues[recallVal] = maxVal
            self.minPrecisionAtFixedRecallValues[recallVal] = minVal
            self.avgPrecisionAtFixedRecallValues[recallVal] = np.average(valuesList)
            self.stdevPrecisionAtFixedRecallValues[recallVal] = np.std(valuesList)
            # tempStdev = valuesList.stream().mapToDouble(a -> Math.pow(a - avgPrecisionAtFixedRecallValues.get(recallVal), 2)).sum();
            # stdevPrecisionAtFixedRecallValues.put(recallVal, Math.sqrt(tempStdev / valuesList.size()));

    def processEntropyBasedMeasures(self, dataset: Dataset):
        IGScoresPerColumnIndex = []
        IGScoresPerDiscreteColumnIndex = []
        IGScoresPerNumericColumnIndex = []

        # start by getting the IG scores of all the attributes
        ige = InformationGainFilterEvaluator()
        trainSet = dataset.getAllColumns(False)
        for idx in trainSet.columns:
            ci = trainSet[idx]
            if dataset.getTargetClassColumn().name == ci.name:
                continue

            # if the attribute is string or date, not much we can do about that
            # if (ci.getColumn().getType() != Column.columnType.Discrete && ci.getColumn().getType() != Column.columnType.Numeric) {
            if (not pd.api.types.is_integer_dtype(ci)) and (not pd.api.types.is_float_dtype(ci)):
                continue

            indicedList = []
            indicedList.append(idx)
            replicatedDataset = dataset.replicateDatasetByColumnIndices(indicedList)
            tempList = []
            tempList.append(ci)
            ige.initFilterEvaluator(tempList)
            score = ige.produceScore(replicatedDataset, None, dataset, None, None)
            IGScoresPerColumnIndex.append(score)

            if pd.api.types.is_integer_dtype(ci):
                IGScoresPerDiscreteColumnIndex.append(score)
            else:
                IGScoresPerNumericColumnIndex.append(score)

    # Used to calculate the dependency of the different attributes in the dataset. For the numeric attributes we conduct a paired T-Test
    # between every pair. For the discrete attributes we conduct a Chi-Square test. Finally, we discretize the numeric attributes and
    # conduct an additional Chi-Suqare test on all attributes.
    def processAttributesStatisticalTests(self, dataset: Dataset):
        pairedTTestValuesList = []
        for i in range(len(self.numericAttributesList)-1):
            for j in range(i+1, len(self.numericAttributesList)):
                if i != j:
                    tTestVal = abs(scipy.stats.ttest_ind(self.numericAttributesList[i], self.numericAttributesList[j]))
                    if not np.isnan(tTestVal) and not np.isinf(tTestVal):
                        pairedTTestValuesList.append(tTestVal)


        if len(pairedTTestValuesList) > 0:
            self.maxPairedTTestValueForNumericAttributes = max(pairedTTestValuesList)
            self.minPairedTTestValueForNumericAttributes = min(pairedTTestValuesList)
            self.avgPairedTTestValueForNumericAttributes = np.average(pairedTTestValuesList)
            self.stdevPairedTTestValueForNumericAttributes = np.std(pairedTTestValuesList)
            # double tempStdev = pairedTTestValuesList.stream().mapToDouble(a -> Math.pow(a - self.avgPairedTTestValueForNumericAttributes, 2)).sum();
            # self.stdevPairedTTestValueForNumericAttributes = Math.sqrt(tempStdev / pairedTTestValuesList.size());

        else:
            self.maxPairedTTestValueForNumericAttributes = 0
            self.minPairedTTestValueForNumericAttributes = 0
            self.avgPairedTTestValueForNumericAttributes = 0
            self.stdevPairedTTestValueForNumericAttributes = 0

        # Next we calculate the Chi-Square TEST OF INDEPENDENCE for the discrete attributes
        # TODO: figure out chi square function
        # ChiSquareTest chiSquareTest = new ChiSquareTest()
        chiSquaredTestValuesList = []
        for i in range(len(self.discreteAttributesList)-1):
            for j in range(i+1, len(self.discreteAttributesList)):
                if i != j:
                    counts = self.generateDiscreteAttributesCategoryIntersection(self.discreteAttributesList[i], self.discreteAttributesList[i])
                    # testVal = chiSquareTest.chiSquare(counts)
                    # if not np.isnan(testVal) and not np.isinf(testVal):
                    #     chiSquaredTestValuesList.append(testVal)

        if len(chiSquaredTestValuesList) > 0:
            self.maxChiSquareValueforDiscreteAttributes = max(chiSquaredTestValuesList)
            self.minChiSquareValueforDiscreteAttributes = min(chiSquaredTestValuesList)
            self.avgChiSquareValueforDiscreteAttributes = np.average(chiSquaredTestValuesList)
            self.stdevChiSquareValueforDiscreteAttributes = np.std(chiSquaredTestValuesList)
            # double tempStdev = chiSquaredTestValuesList.stream().mapToDouble(a -> Math.pow(a - self.avgChiSquareValueforDiscreteAttributes, 2)).sum();
            # self.stdevChiSquareValueforDiscreteAttributes = Math.sqrt(tempStdev / chiSquaredTestValuesList.size());
        else:
            self.maxChiSquareValueforDiscreteAttributes = 0
            self.minChiSquareValueforDiscreteAttributes = 0
            self.avgChiSquareValueforDiscreteAttributes = 0
            self.stdevChiSquareValueforDiscreteAttributes = 0

        # finally, we discretize the numberic features and conduct an additional Chi-Square evaluation
        bins = [0] * Properties.equalRangeDiscretizerBinsNumber
        # EqualRangeDiscretizerUnaryOperator erduo
        discretizedColumns = []
        for ci in self.numericAttributesList:
            erduo = EqualRangeDiscretizerUnaryOperator(bins)
            tempColumnsList = []
            tempColumnsList.append(ci)
            erduo.processTrainingSet(dataset,tempColumnsList,None)
            discretizedAttribute = erduo.generate(dataset, tempColumnsList, None, False)
            discretizedColumns.append(discretizedAttribute)

        # now we add all the original discrete attributes to this list and run the Chi-Square test again
        discretizedColumns.extend(self.discreteAttributesList)
        chiSquaredTestValuesList = []
        for i in range(len(discretizedColumns)-1):
            for j in range(i + 1, len(discretizedColumns)):
                if (i!=j):
                    pass
                    # TODO: same as above
                    # long[][] counts = generateDiscreteAttributesCategoryIntersection((DiscreteColumn) discretizedColumns.get(i).getColumn(), (DiscreteColumn) discretizedColumns.get(j).getColumn());
                    # double testVal = chiSquareTest.chiSquare(counts);
                    # if (!Double.isNaN(testVal) &&  !Double.isInfinite(testVal)) {
                    #     chiSquaredTestValuesList.add(testVal);


        if len(chiSquaredTestValuesList) > 0:
            self.maxChiSquareValueforDiscreteAndDiscretizedAttributes = max(chiSquaredTestValuesList)
            self.minChiSquareValueforDiscreteAndDiscretizedAttributes = min(chiSquaredTestValuesList)
            self.avgChiSquareValueforDiscreteAndDiscretizedAttributes = np.average(chiSquaredTestValuesList)
            self.stdevChiSquareValueforDiscreteAndDiscretizedAttributes = np.std(chiSquaredTestValuesList)
            # double tempStdev = chiSquaredTestValuesList.stream().mapToDouble(a -> Math.pow(a - self.avgChiSquareValueforDiscreteAndDiscretizedAttributes, 2)).sum();
            # self.stdevChiSquareValueforDiscreteAndDiscretizedAttributes = Math.sqrt(tempStdev / chiSquaredTestValuesList.size());

        else:
            self.maxChiSquareValueforDiscreteAndDiscretizedAttributes = 0
            self.minChiSquareValueforDiscreteAndDiscretizedAttributes = 0
            self.avgChiSquareValueforDiscreteAndDiscretizedAttributes = 0
            self.stdevChiSquareValueforDiscreteAndDiscretizedAttributes = 0

    # Returns a HashMap with all the attributes. For each attribute, in addition to the value we also record the
    # type and name of each attribute.
    def generateDatasetAttributesMap(self):
        attributes = {}

        att1 = AttributeInfo("numOfInstances", Column.columnType.Numeric, self.numOfInstances,-1)
        att2 = AttributeInfo("numOfClasses", Column.columnType.Numeric, self.numOfClasses,-1)
        att3 = AttributeInfo("numOfFeatures", Column.columnType.Numeric, self.numOfFeatures,-1)
        att4 = AttributeInfo("numOfNumericAtributes", Column.columnType.Numeric, self.numOfNumericAtributes,-1)
        att5 = AttributeInfo("numOfDiscreteAttributes", Column.columnType.Numeric, self.numOfDiscreteAttributes,-1)
        att6 = AttributeInfo("ratioOfNumericAttributes", Column.columnType.Numeric, self.ratioOfNumericAttributes,-1)
        att7 = AttributeInfo("ratioOfDiscreteAttributes", Column.columnType.Numeric, self.ratioOfDiscreteAttributes,-1)
        att8 = AttributeInfo("maxNumberOfDiscreteValuesPerAttribute", Column.columnType.Numeric, self.maxNumberOfDiscreteValuesPerAttribute,-1)
        att9 = AttributeInfo("minNumberOfDiscreteValuesPerAttribtue", Column.columnType.Numeric, self.minNumberOfDiscreteValuesPerAttribtue,-1)
        att10 = AttributeInfo("avgNumOfDiscreteValuesPerAttribute", Column.columnType.Numeric, self.avgNumOfDiscreteValuesPerAttribute,-1)
        att11 = AttributeInfo("stdevNumOfDiscreteValuesPerAttribute", Column.columnType.Numeric, self.stdevNumOfDiscreteValuesPerAttribute,-1)
        att12 = AttributeInfo("numOfFoldsInEvaluation", Column.columnType.Numeric, self.numOfFoldsInEvaluation,-1)
        att13 = AttributeInfo("maxAUC", Column.columnType.Numeric, self.maxAUC,-1)
        att14 = AttributeInfo("minAUC", Column.columnType.Numeric, self.minAUC,-1)
        att15 = AttributeInfo("avgAUC", Column.columnType.Numeric, self.avgAUC,-1)
        att16 = AttributeInfo("stdevAUC", Column.columnType.Numeric, self.stdevAUC,-1)
        att17 = AttributeInfo("maxLogLoss", Column.columnType.Numeric, self.maxLogLoss,-1)
        att18 = AttributeInfo("minLogLoss", Column.columnType.Numeric, self.minLogLoss,-1)
        att19 = AttributeInfo("avgLogLoss", Column.columnType.Numeric, self.avgLogLoss,-1)
        att20 = AttributeInfo("stdevLogLoss", Column.columnType.Numeric, self.stdevLogLoss,-1)
        att21 = AttributeInfo("maxIGVal", Column.columnType.Numeric, self.maxIGVal,-1)
        att22 = AttributeInfo("minIGVal", Column.columnType.Numeric, self.minIGVal,-1)
        att23 = AttributeInfo("avgIGVal", Column.columnType.Numeric, self.avgIGVal,-1)
        att24 = AttributeInfo("stdevIGVal", Column.columnType.Numeric, self.stdevIGVal,-1)
        att25 = AttributeInfo("discreteAttsMaxIGVal", Column.columnType.Numeric, self.discreteAttsMaxIGVal,-1)
        att26 = AttributeInfo("discreteAttsMinIGVal", Column.columnType.Numeric, self.discreteAttsMinIGVal,-1)
        att27 = AttributeInfo("discreteAttsAvgIGVal", Column.columnType.Numeric, self.discreteAttsAvgIGVal,-1)
        att28 = AttributeInfo("discreteAttsStdevIGVal", Column.columnType.Numeric, self.discreteAttsStdevIGVal,-1)
        att29 = AttributeInfo("numericAttsMaxIGVal", Column.columnType.Numeric, self.numericAttsMaxIGVal,-1)
        att30 = AttributeInfo("numericAttsMinIGVal", Column.columnType.Numeric, self.numericAttsMinIGVal,-1)
        att31 = AttributeInfo("numericAttsAvgIGVal", Column.columnType.Numeric, self.numericAttsAvgIGVal,-1)
        att32 = AttributeInfo("numericAttsStdevIGVal", Column.columnType.Numeric, self.numericAttsStdevIGVal,-1)
        att33 = AttributeInfo("maxPairedTTestValueForNumericAttributes", Column.columnType.Numeric, self.maxPairedTTestValueForNumericAttributes,-1)
        att34 = AttributeInfo("minPairedTTestValueForNumericAttributes", Column.columnType.Numeric, self.minPairedTTestValueForNumericAttributes,-1)
        att35 = AttributeInfo("avgPairedTTestValueForNumericAttributes", Column.columnType.Numeric, self.avgPairedTTestValueForNumericAttributes,-1)
        att36 = AttributeInfo("stdevPairedTTestValueForNumericAttributes", Column.columnType.Numeric, self.stdevPairedTTestValueForNumericAttributes,-1)
        att37 = AttributeInfo("maxChiSquareValueforDiscreteAttributes", Column.columnType.Numeric, self.maxChiSquareValueforDiscreteAttributes,-1)
        att38 = AttributeInfo("minChiSquareValueforDiscreteAttributes", Column.columnType.Numeric, self.minChiSquareValueforDiscreteAttributes,-1)
        att39 = AttributeInfo("avgChiSquareValueforDiscreteAttributes", Column.columnType.Numeric, self.avgChiSquareValueforDiscreteAttributes,-1)
        att40 = AttributeInfo("stdevChiSquareValueforDiscreteAttributes", Column.columnType.Numeric, self.stdevChiSquareValueforDiscreteAttributes,-1)
        att41 = AttributeInfo("maxChiSquareValueforDiscreteAndDiscretizedAttributes", Column.columnType.Numeric, self.maxChiSquareValueforDiscreteAndDiscretizedAttributes,-1)
        att42 = AttributeInfo("minChiSquareValueforDiscreteAndDiscretizedAttributes", Column.columnType.Numeric, self.minChiSquareValueforDiscreteAndDiscretizedAttributes,-1)
        att43 = AttributeInfo("avgChiSquareValueforDiscreteAndDiscretizedAttributes", Column.columnType.Numeric, self.avgChiSquareValueforDiscreteAndDiscretizedAttributes,-1)
        att44 = AttributeInfo("stdevChiSquareValueforDiscreteAndDiscretizedAttributes", Column.columnType.Numeric, self.stdevChiSquareValueforDiscreteAndDiscretizedAttributes,-1)
        att45 = AttributeInfo("minorityClassPercentage", Column.columnType.Numeric, self.minorityClassPercentage,-1)

        attributes[len(attributes)] = att1
        attributes[len(attributes)] = att2
        attributes[len(attributes)] = att3
        attributes[len(attributes)] = att4
        attributes[len(attributes)] = att5
        attributes[len(attributes)] = att6
        attributes[len(attributes)] = att7
        attributes[len(attributes)] = att8
        attributes[len(attributes)] = att9
        attributes[len(attributes)] = att10
        attributes[len(attributes)] = att11
        attributes[len(attributes)] = att12
        attributes[len(attributes)] = att13
        attributes[len(attributes)] = att14
        attributes[len(attributes)] = att15
        attributes[len(attributes)] = att16
        attributes[len(attributes)] = att17
        attributes[len(attributes)] = att18
        attributes[len(attributes)] = att19
        attributes[len(attributes)] = att20
        attributes[len(attributes)] = att21
        attributes[len(attributes)] = att22
        attributes[len(attributes)] = att23
        attributes[len(attributes)] = att24
        attributes[len(attributes)] = att25
        attributes[len(attributes)] = att26
        attributes[len(attributes)] = att27
        attributes[len(attributes)] = att28
        attributes[len(attributes)] = att29
        attributes[len(attributes)] = att30
        attributes[len(attributes)] = att31
        attributes[len(attributes)] = att32
        attributes[len(attributes)] = att33
        attributes[len(attributes)] = att34
        attributes[len(attributes)] = att35
        attributes[len(attributes)] = att36
        attributes[len(attributes)] = att37
        attributes[len(attributes)] = att38
        attributes[len(attributes)] = att39
        attributes[len(attributes)] = att40
        attributes[len(attributes)] = att41
        attributes[len(attributes)] = att42
        attributes[len(attributes)] = att43
        attributes[len(attributes)] = att44
        attributes[len(attributes)] = att45

        # now we need to process the multiple values of the precision/recall analysis.
        for key in self.maxPrecisionAtFixedRecallValues.keys():
            maxPrecisionAtt =  AttributeInfo("maxPrecisionAtFixedRecallValues_" + key, Column.columnType.Numeric, self.maxPrecisionAtFixedRecallValues[key],-1)
            minPrecisionAtt =  AttributeInfo("minPrecisionAtFixedRecallValues_" + key, Column.columnType.Numeric, self.minPrecisionAtFixedRecallValues[key],-1)
            avgPrecisionAtt =  AttributeInfo("avgPrecisionAtFixedRecallValues_" + key, Column.columnType.Numeric, self.avgPrecisionAtFixedRecallValues[key],-1)
            stdevPrecisionAtt =  AttributeInfo("stdevPrecisionAtFixedRecallValues_" + key, Column.columnType.Numeric, self.stdevPrecisionAtFixedRecallValues[key],-1)
            attributes[len(attributes)] = maxPrecisionAtt
            attributes[len(attributes)] = minPrecisionAtt
            attributes[len(attributes)] = avgPrecisionAtt
            attributes[len(attributes)] = stdevPrecisionAtt

        return attributes

    def generateDiscreteAttributesCategoryIntersection(self, col1, col2):
        intersectionsMatrix = np.zeros((col1.getNumOfPossibleValues(), col2.getNumOfPossibleValues()))
        col1Values = col1.getValues()
        col2Values = col2.getValues()

        if len(col1Values) != len(col2Values):
            raise Exception("Columns do not have the same number of instances")

        for i in range(len(col1Values)):
            intersectionsMatrix[col1Values[i]][col2Values[i]] += 1

        return intersectionsMatrix



