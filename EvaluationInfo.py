

from Evaluation import Evaluation

class EvaluationInfo:

    # evaluationStats: Classifier
    # scoreDistributions: 2d array of test predictions
    def __init__(self, evaluationStats, scoreDistributions):
        self.evaluation = evaluationStats
        self.scoreDistPerInstance = scoreDistributions

    def getEvaluationStats(self):
        return self.evaluation

    def getScoreDistribution(self):
        return self.scoreDistPerInstance

