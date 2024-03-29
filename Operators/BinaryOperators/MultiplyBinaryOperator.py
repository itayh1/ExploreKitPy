from typing import List

import numpy as np
import pandas as pd

from Data.Dataset import Dataset
from Evaluation.OperationAssignmentAncestorsSingleton import OperationAssignmentAncestorsSingleton
from Operators.BinaryOperators.BinaryOperator import BinaryOperator
from Operators.Operator import outputType, operatorType


class MultiplyBinaryOperator(BinaryOperator):

    def generate(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns):
        newColumn = sourceColumns[0].mul(targetColumns[0]).replace([np.inf, -np.inf], np.nan).fillna(0)
        newColumn.name = 'Multiply' + self.generateName(sourceColumns, targetColumns)
        oaAncestors = OperationAssignmentAncestorsSingleton()
        oaAncestors.addAssignment(newColumn.name, sourceColumns, targetColumns)
        return newColumn

    def processTrainingSet(self, dataset: Dataset, sourceColumns: pd.Series, targetColumns):
        pass

    def getType(self) -> operatorType:
        return operatorType.Binary

    def getOutputType(self) -> outputType:
        return outputType.Numeric

    def getName(self) -> str:
        return 'MultiplyBinaryOperator'

