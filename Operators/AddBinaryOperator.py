from Dataset import Dataset
from Operators.BinaryOperator import BinaryOperator
from Operators.Operator import outputType, operatorType


class AddBinaryOperator(BinaryOperator):

    def generate(self, dataset: Dataset, sourceColumns, targetColumns):
        return sourceColumns + targetColumns
        # NumericColumn column = new NumericColumn(dataset.getNumOfInstancesPerColumn());
        #
        # int numOfRows = dataset.getNumOfTrainingDatasetRows() + dataset.getNumOfTestDatasetRows();
        # NumericColumn sourceColumn = (NumericColumn)sourceColumns.get(0).getColumn();
        # NumericColumn targetColumn = (NumericColumn)targetColumns.get(0).getColumn();
        #
        # for (int i=0; i<numOfRows; i++) {
        #     int j = dataset.getIndices().get(i);
        #     double val = ((double)sourceColumn.getValue(j)) + ((double)targetColumn.getValue(j));
        #     if (Double.isNaN(val) || Double.isInfinite(val)) {
        #         column.setValue(j, 0);
        #     }
        #     else {
        #         column.setValue(j, val);
        #     }
        # }
        #
        # ColumnInfo newColumnInfo = new ColumnInfo(column, sourceColumns, targetColumns, this.getClass(), "Add" + generateName(sourceColumns,targetColumns));
        # if (enforceDistinctVal && !super.isDistinctValEnforced(dataset,newColumnInfo)) {
        #     return null;
        # }
        # return newColumnInfo;

    def getType(self) -> operatorType:
        return operatorType.Binary

    def getOutputType(self) -> outputType:
        return outputType.Numeric

    def getName(self) -> str:
        return 'AddBinaryOperator'

