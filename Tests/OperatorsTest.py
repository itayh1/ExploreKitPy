from Evaluation.OperatorAssignmentBasedAttributes import OperatorAssignmentBasedAttributes
from Operators.GroupByThenOperators.GroupByThen import GroupByThen
from Operators.GroupByThenOperators.GroupByThenAvg import GroupByThenAvg
from Operators.GroupByThenOperators.GroupByThenCount import GroupByThenCount
from Evaluation.OperatorsAssignmentsManager import OperatorsAssignmentsManager
from Utils.Loader import Loader

if __name__ == '__main__':
    baseFolder = '/home/itay/Documents/java/ExploreKit/AutomaticFeatureGeneration-master/ML_Background/Datasets/'
    german_credit_dataset_path = baseFolder + "german_credit.arff"

    loader = Loader()
    randomSeed = 42
    dataset = loader.readArff(german_credit_dataset_path, randomSeed, None, None, 0.66)

    # nonUnaryOperators = OperatorsAssignmentsManager.getNonUnaryOperatorsList()
    nonUnaryOperators = [GroupByThenCount(), GroupByThenAvg()]
    nonUnaryOperatorsAssignments = OperatorsAssignmentsManager.getOperatorAssignments(dataset, None, nonUnaryOperators,2)

    for oa in nonUnaryOperatorsAssignments:
        operator = OperatorsAssignmentsManager.getOperator(oa.getOperator())
        operator.processTrainingSet(dataset, oa.getSources(), oa.getTargets())
        col = operator.generate(dataset, oa.getSources(), oa.getTargets())
        print(col)