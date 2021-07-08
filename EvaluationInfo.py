

from Evaluation import Evaluation
import numpy as np

class EvaluationInfo:

    # evaluationStats: Classifier
    # scoreDistributions: 2d array of test predictions
    def __init__(self, evaluationStats, scoreDistributions: np.ndarray):
        self.evaluation = evaluationStats
        self.scoreDistPerInstance: np.ndarray = scoreDistributions
        self.predictions:np.ndarray = np.max(scoreDistributions, axis=1)

    def getEvaluationStats(self):
        return self.evaluation

    def getScoreDistribution(self) -> np.ndarray:
        return self.scoreDistPerInstance

    def getPredictions(self) -> np.ndarray:
        return  self.predictions
