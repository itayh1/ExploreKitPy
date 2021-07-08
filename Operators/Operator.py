
from enum import Enum

class operatorType(Enum):
    Unary = 1
    Binary = 2
    GroupByThen = 3
    TimeBasedGroupByThen = 4

class outputType(Enum):
    Numeric = 1
    Discrete = 2
    Date = 3

class Operator:

    def __init__(self):
        pass