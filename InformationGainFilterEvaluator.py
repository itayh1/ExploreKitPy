from ClassificationResults import ClassificationResults
from Dataset import Dataset
from FilterEvaluator import FilterEvaluator
from OperatorAssignment import OperatorAssignment

import math

class InformationGainFilterEvaluator(FilterEvaluator):

    def initFilterEvaluator(self, columnsToAnalyze: list):
        self.valuesPerKey: dict = {} #<List<Integer>, int[]>

    def produceScore(self, analyzedDatasets: Dataset, currentScore: ClassificationResults, completeDataset: Dataset, oa: OperatorAssignment, candidateAttribute):
        if candidateAttribute != None:
            analyzedDatasets.addColumn(candidateAttribute)


        # if any of the analyzed attribute is not discrete, it needs to be discretized
        bins = []
        super().discretizeColumns(analyzedDatasets, bins)
        # Todo: distinct value. make sure to ignore
        # if (analyzedDatasets.getDistinctValueColumns() != None) and (analyzedDatasets.getDistinctValueColumns().size() > 0):
        #     return self.produceScoreWithDistinctValues(analyzedDatasets, currentScore, oa, candidateAttribute)


        valuesPerKey = {}
        targetColumn = analyzedDatasets.getTargetClassColumn()

        # In filter evaluators we evaluate the test set, the same as we do in wrappers. The only difference here is that we
        # train and test on the test set directly, while in the wrappers we train a model on the training set and then apply on the test set
        for  i in range(analyzedDatasets.getNumOfTestDatasetRows()):
            j = analyzedDatasets.getIndicesOfTestInstances().get(i);
            sourceValues: list = [c.getColumn().getValue(j) for c in super().analyzedColumns]
            targetValue = targetColumn.getColumn().getValue(j)
            if sourceValues not in valuesPerKey:
                valuesPerKey[sourceValues] = [0] * analyzedDatasets.getTargetClassColumn().getColumn().getNumOfPossibleValues()
            valuesPerKey.get(sourceValues)[targetValue] += 1

        return self.calculateIG(analyzedDatasets)

    # def produceScoreWithDistinctValues(self, dataset:Dataset , currentScore:ClassificationResults, oa:OperatorAssignment, candidateAttribute:ColumnInfo):
    #     pass

    def calculateIG(self, dataset: Dataset):
        IG = 0.0
        for val in self.valuesPerKey.values():
            numOfInstances = sum(val)
            tempIG = 0
            for value in val:
                if value != 0:
                    tempIG += -((value / numOfInstances) * math.log10(value / numOfInstances))

            IG += (numOfInstances/dataset.getNumOfTrainingDatasetRows()) * tempIG
        return IG

