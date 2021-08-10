from typing import List

from ClassificationResults import ClassificationResults
from Dataset import Dataset
from OperatorAssignment import OperatorAssignment

class PreRankerScoreRanker:

    def rankAndFilter(self, operatorAssignments: List[OperatorAssignment], previousIterationChosenAttributes:List,
                                    datasets:List[Dataset], currentScore:List[ClassificationResults])->List[OperatorAssignment]:
        operatorAssignments.sort(key=lambda x: x.preRankerEvaluatorScore, reverse=True)
        return operatorAssignments