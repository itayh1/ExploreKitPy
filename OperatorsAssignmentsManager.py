from builtins import staticmethod
from typing import List

from ClassificationResults import ClassificationResults
from Dataset import Dataset
from FilterEvaluator import FilterEvaluator
from FilterPreRankerEvaluator import FilterPreRankerEvaluator
from InformationGainFilterEvaluator import InformationGainFilterEvaluator
from OperatorAssignment import OperatorAssignment
from Operators.AddBinaryOperator import AddBinaryOperator
from Operators.CombinationGenerator import CombinationGenerator
from Operators.EqualRangeDiscretizerUnaryOperator import EqualRangeDiscretizerUnaryOperator
from Operators import Operator
from Properties import Properties


class OperatorsAssignmentsManager:


    # Returns a list of unary operators from the configuration file
    @staticmethod
    def getUnaryOperatorsList():
        operatorNames = Properties.unaryOperators.split(",")
        unaryOperatorsList = []
        for unaryOperator in operatorNames:
            uo = OperatorsAssignmentsManager.getUnaryOperator(unaryOperator)
            unaryOperatorsList.append(uo)
        return unaryOperatorsList


    # Returns an unary operator by name
    @staticmethod
    def getUnaryOperator(operatorName: str):
        if operatorName == "EqualRangeDiscretizerUnaryOperator":
            bins = [0] * int(Properties.equalRangeDiscretizerBinsNumber)
            erd = EqualRangeDiscretizerUnaryOperator(bins)
            return erd
        # elif operatorName == "StandardScoreUnaryOperator":
        #     ssuo = StandardScoreUnaryOperator()
        #     return ssuo
        # elif operatorName == "DayOfWeekUnaryOperator":
        #     dowuo = DayOfWeekUnaryOperator()
        #     return dowuo
        # elif operatorName == "HourOfDayUnaryOperator":
        #     hoduo = HourOfDayUnaryOperator()
        #     return hoduo
        # elif operatorName == "IsWeekendUnaryOperator":
        #     iwuo = IsWeekendUnaryOperator()
        #     return iwuo
        else:
            raise Exception("unindentified unary operator: " + operatorName)


    # Returns a list of nonUnary operators from the configuration file (i.e. all other operator types)
    @staticmethod
    def getNonUnaryOperatorsList():
        operatorNames = Properties.nonUnaryOperators.split(',')
        operatorsList = []
        for unaryOperator in operatorNames:
            operator = OperatorsAssignmentsManager.getNonUnaryOperator(unaryOperator)
            operatorsList.append(operator)

        return operatorsList


    # Returns a non-unary operator by name
    @staticmethod
    def getNonUnaryOperator(operatorName: str):
        timeSpan = 0
        if operatorName.startswith("TimeBasedGroupByThen"):
            timeSpan = float(operatorName.split("_")[1])
            operatorName = operatorName.split("_")[0]

        # switch (operatorName) {
        # GroupByThenOperators
        # if operatorName == "GroupByThenAvg":
        #     GroupByThenAvg gbtAvg = new GroupByThenAvg();
        #     return gbtAvg
        # elif operatorName == "GroupByThenMax":
        #     GroupByThenMax gbtMmax = new GroupByThenMax();
        #     return gbtMmax
        # elif operatorName == "GroupByThenMin":
        #     GroupByThenMin gbtMin = new GroupByThenMin();
        #     return gbtMin
        # elif operatorName == "GroupByThenCount":
        #     GroupByThenCount gbtCount = new GroupByThenCount();
        #     return gbtCount
        # elif operatorName == "GroupByThenStdev":
        #     GroupByThenStdev gbtStdev = new GroupByThenStdev();
        #     return gbtStdev

        # BinaryOperators
        if operatorName == "AddBinaryOperator":
            abo = AddBinaryOperator()
            return abo
        # elif operatorName == "SubtractBinaryOperator":
        #     SubtractBinaryOperator sbo = new SubtractBinaryOperator();
        #     return sbo;
        # elif operatorName == "MultiplyBinaryOperator":
        #     MultiplyBinaryOperator mbo = new MultiplyBinaryOperator();
        #     return mbo;
        # elif operatorName == "DivisionBinaryOperator":
        #     DivisionBinaryOperator dbo = new DivisionBinaryOperator();
        #     return dbo;

        # Todo: ignore time features
        # TimeBasedGroupByThen
        # elif operatorName == "TimeBasedGroupByThenCountAndAvg":
        #     TimeBasedGroupByThenCountAndAvg tbgbycaa = new TimeBasedGroupByThenCountAndAvg(timeSpan);
        #     return tbgbycaa;
        # elif operatorName == "TimeBasedGroupByThenCountAndCount":
        #     TimeBasedGroupByThenCountAndCount tbgbtccac = new TimeBasedGroupByThenCountAndCount(timeSpan);
        #     return tbgbtccac;
        # elif operatorName == "TimeBasedGroupByThenCountAndMax":
        #     TimeBasedGroupByThenCountAndMax tbgbtcam = new TimeBasedGroupByThenCountAndMax(timeSpan);
        #     return tbgbtcam;
        # elif operatorName == "TimeBasedGroupByThenCountAndMin":
        #     TimeBasedGroupByThenCountAndMin tbgbtcamm = new TimeBasedGroupByThenCountAndMin(timeSpan);
        #     return tbgbtcamm;
        # elif operatorName == "TimeBasedGroupByThenCountAndStdev":
        #     TimeBasedGroupByThenCountAndStdev tbgbtcas = new TimeBasedGroupByThenCountAndStdev(timeSpan);
        #     return tbgbtcas;
        else:
            raise Exception("unindentified unary operator: " + operatorName)


     # Receives a dataset with a set of attributes and a list of operators and generates all possible source/target/operator/secondary operator assignments
     # @param dataset The dataset with the attributes that need to be analyzed
     # @param attributesToInclude A list of attributes that must be included in either the source or target of every generated assignment. If left empty, there are no restrictions
     # @param operators A list of all the operators whose assignment will be considered
     # @param maxCombinationSize the maximal number of attributes that can be a in the source of each operator. Smaller number (down to 1) are also considered
    @staticmethod
    def getOperatorAssignments(dataset: Dataset, attributesToInclude: list, operators: list, maxCombinationSize: int):
        areNonUniaryOperatorsBeingUsed = False
        if len(operators) > 0 and not operators[0].getType().equals(Operator.operatorType.Unary):
            areNonUniaryOperatorsBeingUsed = True

        if attributesToInclude == None: attributesToInclude = []
        operatorsAssignments = []
        for i in range(maxCombinationSize, 0, -1): # (int i=maxCombinationSize; i>0; i--) {
            # List<List<ColumnInfo>>
            sourceAttributeCombinations = OperatorsAssignmentsManager.getAttributeCombinations(dataset.getAllColumns(False), i)

            # for each of the candidate source attributes combinations
            for sources in sourceAttributeCombinations:
                # if a distinct dolumn(s) exists, we need to make sure that at least one column (or one of its ancestors) satisfies the constraint
                # ignore - distinct value
                # if dataset.getDistinctValueColumns() != None and len(dataset.getDistinctValueColumns()) > 0:
                #     if areNonUniaryOperatorsBeingUsed and not isDistinctValueCompliantAttributeExists(dataset.getDistinctValueCompliantColumns(), sources)) {
                #         continue;
                #     }
                # }

                # first check if any of the required atts (if there are any) are included
                if len(attributesToInclude) > 0:
                    tempList = sources.copy()
                    tempList = [item for item in tempList if item in attributesToInclude]
                    if len(tempList) == 0:
                        continue

                # Now we check all the operators on the source attributes alone.
                for operator in operators:
                    if operator.isApplicable(dataset, sources, []):
                        os = OperatorAssignment(sources, None, getOperator(operator), None)
                        operatorsAssignments.append(os)

                    # now we pair the source attributes with a target attribute and check again
                    for targetColumn in dataset.getAllColumns(False):
                        # if (sources.contains(targetColumn)) { continue; }
                        if OperatorsAssignmentsManager.overlapExistsBetweenSourceAndTargetAttributes(sources,targetColumn): continue
                        tempList = []
                        tempList.append(targetColumn)
                        if operator.isApplicable(dataset, sources, tempList):
                            os = OperatorAssignment(sources, tempList, getOperator(operator), None)
                            operatorsAssignments.append(os)

    @staticmethod
    def overlapExistsBetweenSourceAndTargetAttributes(sourceAtts: list, targetAtt) -> bool:
        # the simplest case - the same attribute appears both in the source and the target
        if (sourceAtts.contains(targetAtt)):
            return True


        # Now we need to check that the source atts and the target att has no shared columns (including after the application of an operator)
        sourceAttsAndAncestors = []
        for sourceAtt in sourceAtts:
            sourceAttsAndAncestors.append(sourceAtt)
            if sourceAtt.getSourceColumns() != None:
                for ancestorAtt in sourceAtt.getSourceColumns():
                    # if (!sourceAttsAndAncestors.contains(ancestorAtt)) {
                    if ancestorAtt not in sourceAttsAndAncestors:
                        sourceAttsAndAncestors.append(ancestorAtt)

            if sourceAtt.getTargetColumns() != None:
                for ancestorAtt in sourceAtt.getTargetColumns():
                    # if (!sourceAttsAndAncestors.contains(ancestorAtt)) {
                    if ancestorAtt not in sourceAttsAndAncestors:
                        sourceAttsAndAncestors.append(ancestorAtt)

        # do the same for the target att (because we only have one we don't need the external loop)
        targetAttsAndAncestors = []
        targetAttsAndAncestors.append(targetAtt)
        if targetAtt.getSourceColumns() != None:
            for ancestorAtt in targetAtt.getSourceColumns():
                # if (!targetAttsAndAncestors.contains(ancestorAtt)) {
                if ancestorAtt not in targetAttsAndAncestors:
                    targetAttsAndAncestors.append(ancestorAtt)

        if targetAtt.getTargetColumns() != None:
            for ancestorAtt in targetAtt.getTargetColumns():
                # if (!targetAttsAndAncestors.contains(ancestorAtt)) {
                if ancestorAtt not in targetAttsAndAncestors:
                    targetAttsAndAncestors.append(ancestorAtt)

        # boolean overlap =  !Collections.disjoint(sourceAttsAndAncestors, targetAttsAndAncestors);
        if len(set(sourceAttsAndAncestors).intersection(set(targetAttsAndAncestors))) > 0:
            overlap =  True
        else:
            overlap = False

        #Todo: what is it mean
        if overlap and len(targetAttsAndAncestors) > 1:
            x=5

        return overlap

    # Returns lists of column-combinations
    # @param attributes
    # @param numOfAttributesInCombination
    @staticmethod
    def getAttributeCombinations(self, attributes: list, numOfAttributesInCombination: int) -> list:
        attributeCombinations = []
        gen = CombinationGenerator(len(attributes), numOfAttributesInCombination)
        while gen.hasMore():
            indices = gen.getNext()
            tempColumns = []
            for index in indices:
                tempColumns.append(attributes[index])
            attributeCombinations.append(tempColumns)

        return attributeCombinations


    # Activates the applyOperatorsAndPerformInitialEvaluation function, but only for Unary Operators
    # @param dataset
    # @param mustIncluseAttributes Attributes which must be in either the source or the target of every generated feature
    @staticmethod
    def applyUnaryOperators(dataset: Dataset, mustIncluseAttributes, filterEvaluator: FilterEvaluator,
                            subFoldTrainingDatasets: List[Dataset], currentScores:List[ClassificationResults] ) -> List[OperatorAssignment]:
        unaryOperatorsList = OperatorsAssignmentsManager.getUnaryOperatorsList()
        return OperatorsAssignmentsManager.applyOperatorsAndPerformInitialEvaluation(dataset, unaryOperatorsList,mustIncluseAttributes, 1, filterEvaluator, null, subFoldTrainingDatasets, currentScores, false);


     # Receives a a dataset and a list of operators, finds all possible combinations, generates and writes the attributes to file
     # and returns the assignments list
     # @param dataset The full dataset. The new attribute generated for it is the one to be saved to file
     # @param operators The operators for which assignments will be generated
     # @param mustIncluseAttributes The attributes that must be present in EITHER the source or the target. Empty lists or null mean there's no restriction
     # @param maxNumOfSourceAttributes The maximal number of attributes that can be in the source (if the operator permits). Smaller number down to 1 (including) will also be generated
     # @param filterEvaluator The filter evaluator that will be used to compute the initial ranking of the attriubte. The calculation is carried out on the sibfolds
     # @param preRankerEvaluator
     # @param subFoldTrainingDatasets The training set sub-folds. Used in order to calculate the score, as the test set cannot be used for this purpose here.
    @staticmethod
    def applyOperatorsAndPerformInitialEvaluation(dataset: Dataset, operators: List[Operator], mustIncluseAttributes,
                maxNumOfSourceAttributes: int, filterEvaluator: FilterEvaluator, preRankerEvaluator: FilterPreRankerEvaluator,
                subFoldTrainingDatasets:List[Dataset], currentScores:List[ClassificationResults], reduceNumberOfAttributes:bool) -> List[OperatorAssignment]:

        # in case the number of initial attributes is very high, we need narrow the search space
        if (reduceNumberOfAttributes and (mustIncluseAttributes == None or len(mustIncluseAttributes) == 0)):
            # It is important to break the condition in two, because in advanced interations we always have a "must include" attribute
            if dataset.getAllColumns(False).shape[1] > 60:
                initialSelectionAttEvaluator = InformationGainFilterEvaluator()
                # mustIncluseAttributes = getTopRankingDiscreteAttributesByFilterScore(dataset, initialSelectionAttEvaluator, 10)

        operatorAssignments = OperatorsAssignmentsManager.getOperatorAssignments(dataset, mustIncluseAttributes, operators, maxNumOfSourceAttributes)
        if preRankerEvaluator != None:
            preRankedAttributesToGenerate = Properties.preRankedAttributesToGenerate
            # operatorAssignments = getTopRankingOperatorAssignmentsWithoutGenerating( subFoldTrainingDatasets, operatorAssignments, preRankerEvaluator, preRankedAttributesToGenerate )

        # Create all the new features, save them to file and evaluate them using the filter evaluator
        # generateAttributeAndCalculateFilterEvaluatorScore(dataset, filterEvaluator, subFoldTrainingDatasets, currentScores, operatorAssignments);

        # /*
        # // The single thread version
        # for (OperatorAssignment os: operatorAssignments) {
        #     ColumnInfo ci = generateColumn(dataset, os, true);
        #     //if the filter evaluator is not null, we'll conduct the initial evaluation of the new attribute
        #     if (filterEvaluator != null) {
        #         os.setFilterEvaluatorScore(EvaluateAttributeUsingTrainingSubFolds(subFoldTrainingDatasets, filterEvaluator, os));
        #     }
        # }*/
        return operatorAssignments;
