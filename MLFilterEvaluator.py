

from sklearn.ensemble import RandomForestClassifier

from MLAttributeManager import MLAttributeManager
from Dataset import Dataset
from Logger import Logger


class MLFilterEvaluator:

    analyzedColumns = []
    datasetAttributes = {}

    def __init__(self, dataset: Dataset):

        self.initializeBackgroundModel(dataset)

    def initializeBackgroundModel(self, dataset: Dataset):
        Logger.Info('Initializing background model for dataset' + dataset.name)
        mlam = MLAttributeManager()
        classifier = mlam.getBackgroundClassificationModel(dataset, True)

