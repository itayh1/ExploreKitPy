from Dataset import Dataset


class FilterEvaluator:
#     analyzedColumns = []

    def __init__(self):
        self.analyzedColumns: list = []

    def discretizeColumns(self, dataset: Dataset, bins: list):
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
