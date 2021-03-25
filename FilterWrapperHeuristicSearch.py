

from Properties import Properties
from Dataset import Dataset
from Logger import Logger

class FilterWrapperHeuristicSearch:
    date = None
    maxIteration = 20

    chosenOperatorAssignment = None
    topRankingAssignment = None
    evaluatedAttsCounter = 0

    def __init__(self, maxIterations: int):
        self.maxIteration = maxIterations

    def run(self, originalDataset: Dataset, runInfo: str):
        Logger.Info('Initializing evaluators')

    def getWrapper(self, param):
        pass