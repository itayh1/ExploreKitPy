from Dataset import Dataset
from Logger import Logger
from Operators.Operator import Operator, operatorType, outputType

import numpy as np
import pandas as pd


class EqualRangeDiscretizerUnaryOperator(Operator):

    upperBoundPerBin: list

    def __init__(self, upperBoundPerBin: list):
        self.upperBoundPerBin = upperBoundPerBin

    def processTrainingSet(self, dataset: Dataset, sourceColumns: list, targetColumns:list):
        # minVal = Double.MAX_VALUE;
        # double maxVal = Double.MIN_VALUE;

        columnInfo = sourceColumns.get(0)
        val = columnInfo.getColumn().getValue(0)
        minVal = maxVal = val
        for i in range(dataset.getNumOfTrainingDatasetRows()):
            # TODO: j instead of i
            j = dataset.getIndicesOfTrainingInstances().get(i)
            val = columnInfo.getColumn().getValue(j)
            if (not np.isnan(val)) and (not np.isinf(val)):
                minVal = min(minVal, val)
                maxVal = max(maxVal, val)
            else:
                x=5

        rng = (maxVal-minVal)/len(self.upperBoundPerBin)
        currentVal = minVal
        for i in range(len(self.upperBoundPerBin)):
            self.upperBoundPerBin[i] = currentVal + rng
            currentVal += rng

    def generate(self, dataset: Dataset, sourceColumns: list, targetColumns: list, enforceDistinctVal: bool):
        try:
            # DiscreteColumn column = DiscreteColumn(dataset.getNumOfInstancesPerColumn(), upperBoundPerBin.length)
            column = np.empty(dataset.getNumOfInstancesPerColumn(), dtype=np.dtype(int))
            # this is the number of rows we need to work on - not the size of the vector
            numOfRows = dataset.getNumberOfRows()
            columnInfo = sourceColumns.get(0)
            for i in range(numOfRows):
                # if (dataset.getIndices().size() == i) {
                #     int x = 5;
                # }
                j = dataset.getIndices()[i]
                binIndex = self.GetBinIndex(columnInfo.getColumn().getValue(j))
                column.setValue(j, binIndex)

            # now we generate the name of the new attribute
            attString = "EqualRangeDiscretizer(" + columnInfo.getName()+ ")"
            return pd.DataFrame({attString: column})
            # new ColumnInfo(column, sourceColumns, targetColumns, this.getClass(), attString);

        except Exception as ex:
            Logger.Error("error in EqualRangeDiscretizer:  " +  ex)
            return None

    def GetBinIndex(self, value: float):
        for i in range(len(self.upperBoundPerBin)):
            if self.upperBoundPerBin[i] > value:
                return i

        return len(self.upperBoundPerBin) - 1

    def getType(self) -> operatorType:
        return operatorType.Unary

    def getOutputType(self) -> outputType:
        return outputType.Numeric

    def getName(self) -> str:
        return 'EqualRangeDiscretizerUnaryOperator'


