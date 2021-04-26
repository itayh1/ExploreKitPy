from Dataset import Dataset
from EqualRangeDiscretizerUnaryOperator import EqualRangeDiscretizerUnaryOperator
from Operator import Operator
from Properties import Properties


class OperatorsAssignmentsManager:

    @staticmethod
    # Returns a list of unary operators from the configuration file
    def getUnaryOperatorsList():
        operatorNames = Properties.unaryOperators.split(",")
        unaryOperatorsList = []
        for unaryOperator in operatorNames:
            uo = OperatorsAssignmentsManager.getUnaryOperator(unaryOperator)
            unaryOperatorsList.append(uo)
        return unaryOperatorsList

    @staticmethod
    # Returns an unary operator by name
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
            sourceAttributeCombinations = getAttributeCombinations(dataset.getAllColumns(false), i)

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
                    if len(tempList) == 0: continue


                # Now we check all the operators on the source attributes alone.
                for operator in operators:
                    if operator.isApplicable(dataset, sources, []):
                        os = OperatorAssignment(sources, None, getOperator(operator), None)
                        operatorsAssignments.append(os)

                    # now we pair the source attributes with a target attribute and check again
                    for targetColumn in dataset.getAllColumns(False):
                        # if (sources.contains(targetColumn)) { continue; }
                        if overlapExistsBetweenSourceAndTargetAttributes(sources,targetColumn): continue
                        tempList = []
                        tempList.append(targetColumn)
                        if operator.isApplicable(dataset, sources, tempList):
                            os = OperatorAssignment(sources, tempList, getOperator(operator), None)
                            operatorsAssignments.add(os)

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


