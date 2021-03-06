import pandas as pd

from ClassificationResults import ClassificationResults
from Dataset import Dataset
from OperatorAssignment import OperatorAssignment
from Operators.EqualRangeDiscretizerUnaryOperator import EqualRangeDiscretizerUnaryOperator

class FilterEvaluator:
#     analyzedColumns = []

    def __init__(self):
        self.analyzedColumns: pd.DataFrame

    def initFilterEvaluator(self, analyzedColumns: pd.DataFrame):
        self.analyzedColumns = analyzedColumns

    def discretizeColumns(self, dataset: Dataset, bins: list):
        for colName, col in self.analyzedColumns.iteritems():
            if pd.api.types.is_integer_dtype(col):
                continue
            discretizer = EqualRangeDiscretizerUnaryOperator(bins)
            discretizer.processTrainingSet(dataset, col, None)
            dataset.df[colName] = discretizer.generate(dataset, col, None)
        # for (int i=0; i<analyzedColumns.size(); i++) {
        #     ColumnInfo ci = analyzedColumns.get(i);
        #     if (!ci.getColumn().getType().equals(Column.columnType.Discrete)) {
        #         EqualRangeDiscretizerUnaryOperator  discretizer = new EqualRangeDiscretizerUnaryOperator(bins);
        #         List<ColumnInfo> columns = new ArrayList<>();
        #         columns.add(ci);
        #         discretizer.processTrainingSet(dataset, columns, null);
        #         analyzedColumns.set(i, discretizer.generate(dataset, columns, null, false));
        #     }
        pass


    def produceScore(self, analyzedDatasets: Dataset, currentScore: ClassificationResults, completeDataset: Dataset,
                 oa: OperatorAssignment, candidateAttribute):
        raise NotImplementedError('FilterEvaluator is abstract, must be overrided')

