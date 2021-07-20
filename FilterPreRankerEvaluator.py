from Classifier import Classifier
from Dataset import Dataset
from DatasetBasedAttributes import DatasetBasedAttributes
from Logger import Logger
from MLAttributeManager import MLAttributeManager
from Properties import Properties


class FilterPreRankerEvaluator:

    def __init__(self, dataset: Dataset):
        classifier, datasetAttributes = self.initializeBackgroundModel(dataset)
        self.classifier: Classifier = classifier
        self.datasetAttributes: dict = datasetAttributes

    def initializeBackgroundModel(self, dataset: Dataset):
        Logger.Info("Initializing background model for pre-ranking process")
        mlam = MLAttributeManager()
        classifier = mlam.getBackgroundClassificationModel(dataset, False)

        dba = DatasetBasedAttributes()
        datasetAttributes = dba.getDatasetBasedFeatures(dataset, Properties.classifier)
        return classifier, datasetAttributes
