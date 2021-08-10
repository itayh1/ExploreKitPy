from typing import List

from Operators.Operator import Operator, operatorType


class BinaryOperator(Operator):

    def __init__(self):
        super().__init__()
        self.abc: List[int]

    def getType(self) -> operatorType:
        return operatorType.Unary
