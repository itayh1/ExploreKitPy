
from typing import List

from Operators.Operator import Operator, operatorType, outputType


class UnaryOperator(Operator):

    def __init__(self):
        super().__init__()
        self.abc: List[int]

    def getType(self) -> operatorType:
        return operatorType.Unary

    def requiredInputType(self) -> outputType:
        raise NotImplementedError("UnaryOperator is an abstract class")
