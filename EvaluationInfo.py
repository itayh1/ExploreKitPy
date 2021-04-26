

from Evaluation import Evaluation

class EvaluationInfo:

    def __init__(self,  evaluationStats:Evaluation, scoreDistributions: list):
        self.evaluation = evaluationStats
        self.scoreDistPerInstance = scoreDistributions

