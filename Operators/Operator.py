
from enum import Enum

import pandas as pd

from Dataset import Dataset


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

    def getName(self) -> str:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    def getType(self) -> operatorType:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    def getOutputType(self) -> outputType:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    def processTrainingSet(self, dataset: Dataset, sourceColumns: pd.Series, targetColumns):
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    def generate(self, dataset: Dataset, sourceColumns: pd.Series, targetColumns: list) -> pd.Series:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")