from typing import List

import pandas as pd

from Dataset import Dataset
from Operators.Operator import Operator, operatorType, outputType


class BinaryOperator(Operator):

    def __init__(self):
        super().__init__()
        self.abc: List[int]

    def getType(self) -> operatorType:
        return operatorType.Unary

    def isApplicable(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns: List[pd.Series]) -> bool:
        # if there are any target columns or if there is more than one source column, return false
        if (len(sourceColumns) != 1) or (targetColumns is None) or (len(targetColumns) != 1):
            return False
        if Operator.getSeriesType(sourceColumns[0]) != outputType.Numeric or \
                Operator.getSeriesType(targetColumns[0]) != outputType.Numeric:
            return False
        return True
