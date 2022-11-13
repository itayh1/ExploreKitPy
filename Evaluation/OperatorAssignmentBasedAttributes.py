from typing import Dict, List

import pandas as pd
import numpy as np

from AttributeInfo import AttributeInfo
from Data.Dataset import Dataset
from Evaluation.InformationGainFilterEvaluator import InformationGainFilterEvaluator
from OperatorAssignment import OperatorAssignment
from Operators import Operator

class OperatorAssignmentBasedAttributes:

    def __init__(self):
        self.numOfSources: int
        self.numOfNumericSources: int
        self.numOfDiscreteSources: int
        self.numOfDateSources: int
        self.operatorTypeIdentifier: int # The type of the operator: unary, binary etc.
        self.operatorIdentifier: int
        self.discretizerInUse: int # 0 if none is used, otherwise the type of the discretizer (enumerated) TODO: check if this is applies before or after the operator itself
        self.normalizerInUse: int # 0 if none is used, otherwise the type of the normalizer (enumerated) TODO: check if this is applies before or after the operator itself

        # statistics on the values of discrete source attributes
        self.maxNumOfDiscreteSourceAttribtueValues: float
        self.minNumOfDiscreteSourceAttribtueValues: float
        self.avgNumOfDiscreteSourceAttribtueValues: float
        self.stdevNumOfDiscreteSourceAttribtueValues: float

        # atatistics on the values of the target attribute (currently for numeric values)
        self.maxValueOfNumericTargetAttribute: float
        self.minValueOfNumericTargetAttribute: float
        self.avgValueOfNumericTargetAttribute: float
        self.stdevValueOfNumericTargetAttribute: float

        # statistics on the value of the numeric source attribute (currently we only support cases where it's the only source attribute)
        self.maxValueOfNumericSourceAttribute: float
        self.minValueOfNumericSourceAttribute: float
        self.avgValueOfNumericSourceAttribute: float
        self.stdevValueOfNumericSourceAttribute: float

        # Paired-T amd Chi-Square tests on the source and target attributes
        self.chiSquareTestValueForSourceAttributes: float
        self.pairedTTestValueForSourceAndTargetAttirbutes: float  # used for numeric single source attirbute and numeric target

        self.maxChiSquareTsetForSourceAndTargetAttributes: float # we discretize all the numeric attributes for this one
        self.minChiSquareTsetForSourceAndTargetAttributes: float
        self.avgChiSquareTsetForSourceAndTargetAttributes: float
        self.stdevChiSquareTsetForSourceAndTargetAttributes: float

        # Calculate the similarity of the source attributes to other attibures in the dataset (discretuze all the numeric ones)
        self.maxChiSquareTestvalueForSourceDatasetAttributes: float
        self.minChiSquareTestvalueForSourceDatasetAttributes: float
        self.avgChiSquareTestvalueForSourceDatasetAttributes: float
        self.stdevChiSquareTestvalueForSourceDatasetAttributes: float

        ##########################################################
        # statistics on the generated attributes
        self.isOutputDiscrete: int #if  not, it's 0
        # If the generated attribute is discrete, count the number of possible values. If numeric, the value is set to 0
        self.numOfDiscreteValues: int

        self.IGScore: float

        # If the generated attribute is numeric, calculate the Paired T-Test statistics for it and the datasets's numeric attributes
        self.maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes: float = -1
        self.minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes: float = -1
        self.avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes: float = -1
        self.stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes: float = -1

        # The Chi-Squared test of the (discretized if needed) generate attribute and the dataset's discrete attributes
        self.maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes: float
        self.minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes: float
        self.avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes: float
        self.stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes: float

        # the Chi-Squared test of the (discretized if needed) generate attribute and ALL of the dataset's attributes (discrete and numeric)
        self.maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes: float
        self.minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes: float
        self.avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes: float
        self.stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes: float

        ##########################################################################

        self.probDiffScoreForTopMiscallasiffiedInstancesInMinorityClass: Dict[float, float]
        self.probDiffScoreForTopMiscallasiffiedInstancesInMajorityClass: Dict[float, float]


    # Generates the meta-feautres for the "parents" of the generated attribute. These are the meta-features that DO NOT require
    # calculating the values of the generated attribute to be calculated
    def getOperatorAssignmentBasedMetaFeatures(self, dataset: Dataset, oa: OperatorAssignment) -> Dict[int,AttributeInfo]:
        try:
            # Calling the procedures that calculate the attributes of the OperatorAssignment obejct and the source and target attribtues
            try:
                self.processOperatorAssignment(dataset, oa)
            except Exception as ex:
                x=5

            try:
                self.processSourceAndTargetAttributes(dataset, oa)
            except Exception as ex:
                x = 5

            try:
                self.performStatisticalTestsOnSourceAndTargetAttributes(dataset, oa)
            except Exception as ex:
                x = 5

            try:
                self.performStatisticalTestOnOperatorAssignmentAndDatasetAtributes(dataset, oa)
            except Exception as ex:
                x = 5

            return self.generateInstanceAttributesMap(True, False)

        except Exception as ex:
            return None

    # Generates the meta-features that require the values of the generated attribute in order to be calculated.
    def getGeneratedAttributeValuesMetaFeatures(self, dataset: Dataset, oa: OperatorAssignment, generatedAttribute: pd.Series) -> Dict[int, AttributeInfo]:
        try:
            datasetReplica = dataset.replicateDataset()
            datasetReplica.addColumn(generatedAttribute)
            tempList = []
            tempList.append(generatedAttribute)

            # IGScore
            try:
                igfe = InformationGainFilterEvaluator()
                igfe.initFilterEvaluator(tempList)
                self.IGScore = igfe.produceScore(datasetReplica, None, dataset, None, None)
            except Exception:
                x = 5

            # Calling the procedures that calculate statistics on the candidate attribute
            try:
                self.processGeneratedAttribute(dataset, oa, generatedAttribute)
            except Exception:
                x = 5

            return self.generateInstanceAttributesMap(False, True)
        except Exception:
            return None

    # @param addValuesFreeMetaFeatures If true, will add all the meta-feautres that are not reliant on the values of the generated attribute (i.e. they rely on the "parents" and the operator assignment)
    # @param addValueDependentMetaFeatures If true, will add all the meta-features that are reliant on the values of the generated attribute
    def generateInstanceAttributesMap(self, addValuesFreeMetaFeatures: bool, addValueDependentMetaFeatures: bool) -> Dict[int,AttributeInfo]:
        attributes: Dict[int,AttributeInfo] = {}

        if addValuesFreeMetaFeatures:
            try:
                attributes[len(attributes)] = AttributeInfo("numOfSources", Operator.outputType.Numeric, self.numOfSources, -1)
                attributes[len(attributes)] = AttributeInfo("numOfNumericSources", Operator.outputType.Numeric, self.numOfNumericSources, -1)
                attributes[len(attributes)] = AttributeInfo("numOfDiscreteSources", Operator.outputType.Numeric, self.numOfDiscreteSources, -1)
                attributes[len(attributes)] = AttributeInfo("numOfDateSources", Operator.outputType.Numeric, self.numOfDateSources, -1)
                attributes[len(attributes)] = AttributeInfo("operatorTypeIdentifier", Operator.outputType.Discrete, self.operatorTypeIdentifier, 4)
                attributes[len(attributes)] = AttributeInfo("operatorIdentifier", Operator.outputType.Discrete, self.operatorIdentifier, 30)
                attributes[len(attributes)] = AttributeInfo("discretizerInUse", Operator.outputType.Discrete, self.discretizerInUse, 2)
                attributes[len(attributes)] = AttributeInfo("normalizerInUse", Operator.outputType.Discrete, self.normalizerInUse, 2)
                attributes[len(attributes)] = AttributeInfo("maxNumOfDiscreteSourceAttribtueValues", Operator.outputType.Numeric, self.maxNumOfDiscreteSourceAttribtueValues, -1)
                attributes[len(attributes)] = AttributeInfo("minNumOfDiscreteSourceAttribtueValues", Operator.outputType.Numeric, self.minNumOfDiscreteSourceAttribtueValues, -1)
                attributes[len(attributes)] = AttributeInfo("avgNumOfDiscreteSourceAttribtueValues", Operator.outputType.Numeric, self.avgNumOfDiscreteSourceAttribtueValues, -1)
                attributes[len(attributes)] = AttributeInfo("stdevNumOfDiscreteSourceAttribtueValues", Operator.outputType.Numeric, self.stdevNumOfDiscreteSourceAttribtueValues, -1)
                attributes[len(attributes)] = AttributeInfo("maxValueOfNumericTargetAttribute", Operator.outputType.Numeric, self.maxValueOfNumericTargetAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("minValueOfNumericTargetAttribute", Operator.outputType.Numeric, self.minValueOfNumericTargetAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("avgValueOfNumericTargetAttribute", Operator.outputType.Numeric, self.avgValueOfNumericTargetAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("stdevValueOfNumericTargetAttribute", Operator.outputType.Numeric, self.stdevValueOfNumericTargetAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("maxValueOfNumericSourceAttribute", Operator.outputType.Numeric, self.maxValueOfNumericSourceAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("minValueOfNumericSourceAttribute", Operator.outputType.Numeric, self.minValueOfNumericSourceAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("avgValueOfNumericSourceAttribute", Operator.outputType.Numeric, self.avgValueOfNumericSourceAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("stdevValueOfNumericSourceAttribute", Operator.outputType.Numeric, self.stdevValueOfNumericSourceAttribute, -1)
                attributes[len(attributes)] = AttributeInfo("chiSquareTestValueForSourceAttributes", Operator.outputType.Numeric, self.chiSquareTestValueForSourceAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("pairedTTestValueForSourceAndTargetAttirbutes", Operator.outputType.Numeric, self.pairedTTestValueForSourceAndTargetAttirbutes, -1)
                attributes[len(attributes)] = AttributeInfo("maxChiSquareTsetForSourceAndTargetAttributes", Operator.outputType.Numeric, self.maxChiSquareTsetForSourceAndTargetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("minChiSquareTsetForSourceAndTargetAttributes", Operator.outputType.Numeric, self.minChiSquareTsetForSourceAndTargetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("avgChiSquareTsetForSourceAndTargetAttributes", Operator.outputType.Numeric, self.avgChiSquareTsetForSourceAndTargetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("stdevChiSquareTsetForSourceAndTargetAttributes", Operator.outputType.Numeric, self.stdevChiSquareTsetForSourceAndTargetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("maxChiSquareTestvalueForSourceDatasetAttributes", Operator.outputType.Numeric, self.maxChiSquareTestvalueForSourceDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("minChiSquareTestvalueForSourceDatasetAttributes", Operator.outputType.Numeric, self.minChiSquareTestvalueForSourceDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("avgChiSquareTestvalueForSourceDatasetAttributes", Operator.outputType.Numeric, self.avgChiSquareTestvalueForSourceDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("stdevChiSquareTestvalueForSourceDatasetAttributes", Operator.outputType.Numeric, self.stdevChiSquareTestvalueForSourceDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("isOutputDiscrete", Operator.outputType.Discrete, self.isOutputDiscrete, 2)
                attributes[len(attributes)] = AttributeInfo("numOfDiscreteValues", Operator.outputType.Numeric, self.numOfDiscreteValues, -1) # TODO: in the future, this one will have to move to the other group

            except Exception:
                x = 5

        if addValueDependentMetaFeatures:
            try:
                attributes[len(attributes)] = AttributeInfo("IGvalue", Operator.outputType.Numeric, self.IGScore, -1)
                attributes[len(attributes)] = AttributeInfo("probDiffScore", Operator.outputType.Numeric, -1, -1)
                attributes[len(attributes)] = AttributeInfo("maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Operator.outputType.Numeric, self.maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Operator.outputType.Numeric, self.minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Operator.outputType.Numeric, self.avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes", Operator.outputType.Numeric, self.stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Operator.outputType.Numeric, self.maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Operator.outputType.Numeric, self.minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Operator.outputType.Numeric, self.avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes", Operator.outputType.Numeric, self.stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Operator.outputType.Numeric, self.maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Operator.outputType.Numeric, self.minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Operator.outputType.Numeric, self.avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1)
                attributes[len(attributes)] = AttributeInfo("stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes", Operator.outputType.Numeric, self.stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes, -1)

            except Exception:
                x = 5

        return attributes

    # Used to calculate statistics on the correlation of the generates attribute and the attributes of the dataset.
    # The attributes that were used to generate the feature are excluded.
    def processGeneratedAttribute(self, dataset: Dataset, oa: OperatorAssignment, generatedAttribute: pd.Series):
        # IMPORTANT: make sure that the source and target attributes are not included in these calculations
        discreteColumns: List[pd.Series] = dataset.getAllColumnsOfType(Operator.outputType.Discrete, False)
        numericColumns: List[pd.Series] = dataset.getAllColumnsOfType(Operator.outputType.Numeric, False)

        # The paired T-Tests for the dataset's numeric attributes
        if Operator.Operator.getSeriesType(generatedAttribute) == Operator.outputType.Numeric:
            pairedTTestScores = statisticOperations.calculatePairedTTestValues(filterOperatorAssignmentAttributes(numericColumns, oa), generatedAttribute)
            if len(pairedTTestScores) > 0:
                self.maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = np.max(pairedTTestScores)
                self.minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = np.min(pairedTTestScores)
                self.avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = np.mean(pairedTTestScores)
                self.stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = np.std(pairedTTestScores)
            else:
                self.maxPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0
                self.minPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0
                self.avgPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0
                self.stdevPairedTTestValuesForGeneratedAttributeAndDatasetNumericAttributes = 0

        # The chi Squared test for the dataset's dicrete attribtues
        chiSquareTestsScores = statisticOperations.calculateChiSquareTestValues(filterOperatorAssignmentAttributes(discreteColumns,oa),generatedAttribute,dataset)
        if len(chiSquareTestsScores) > 0:
            self.maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = np.max(chiSquareTestsScores)
            self.minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = np.min(chiSquareTestsScores)
            self.avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = np.mean(chiSquareTestsScores)
            self.stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = np.std(chiSquareTestsScores)
        else:
            self.maxChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0
            self.minChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0
            self.avgChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0
            self.stdevChiSquaredTestValuesForGeneratedAttributeAndDatasetDiscreteAttributes = 0

        # The Chi Square test for ALL the dataset's attirbutes (Numeric attributes will be discretized)
        discreteColumns.extend(numericColumns)
        AllAttributesChiSquareTestsScores = statisticOperations.calculateChiSquareTestValues(filterOperatorAssignmentAttributes(discreteColumns,oa),generatedAttribute,dataset)
        if len(AllAttributesChiSquareTestsScores) > 0:
            self.maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = np.max(AllAttributesChiSquareTestsScores)
            self.minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = np.min(AllAttributesChiSquareTestsScores)
            self.avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = np.mean(AllAttributesChiSquareTestsScores)
            self.stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = np.std(AllAttributesChiSquareTestsScores)
        else:
            self.maxChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = 0
            self.minChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = 0
            self.avgChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes = 0
            self.stdevChiSquaredTestValuesForGeneratedAttributeAndAllDatasetAttributes =  0



    def getNumOfNewAttributeDiscreteValues(self, oa: OperatorAssignment) -> int:
        # TODO: the current code assumes that the only way for a generated attribute to have discrete values is to be generated using the Discretizer unary operator. This will have to be modified as we expand the system. Moreover, we will have to see if this can remain in the part of the code that calculates the meta-features without generating the attributes values

        if oa.getSecondaryOperator() is not None:
            return oa.getSecondaryOperator().getNumOfBins()
        else:
            if oa.getOperator().getOutputType() != Operator.outputType.Discrete:
                return -1
            else:
                # currently the only operators which return a discrete value are the Unary.
                return (oa.getOperator()).getNumOfBins()



