from typing import List

import pandas as pd

from Data.Dataset import Dataset
from Evaluation.OperationAssignmentAncestorsSingleton import OperationAssignmentAncestorsSingleton
from Operators.BinaryOperator import BinaryOperator
from Operators.Operator import outputType, operatorType


class AddBinaryOperator(BinaryOperator):

    def generate(self, dataset: Dataset, sourceColumns: List[pd.Series], targetColumns):
        newColumn = sourceColumns[0].add(targetColumns[0], fill_value=0)
        newColumn.name = 'Add' + self.generateName(sourceColumns, targetColumns)
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
        return 'AddBinaryOperator'

