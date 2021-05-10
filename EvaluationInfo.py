

from Evaluation import Evaluation

class EvaluationInfo:

    def __init__(self,  evaluationStats, scoreDistributions: list):
        self.evaluation = evaluationStats
        self.scoreDistPerInstance = scoreDistributions

