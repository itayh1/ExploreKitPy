

from DatasetBasedAttributes import DatasetBasedAttributes
from MLAttributeManager import MLAttributeManager
from Dataset import Dataset
from Logger import Logger
from Properties import Properties


class MLFilterEvaluator:

    analyzedColumns = []
    datasetAttributes = {}

    def __init__(self, dataset: Dataset):

        self.initializeBackgroundModel(dataset)

    # Used to create or load the data used by the background model - all the datasets that are evaluated "offline" to create
    # the meta-features classifier.
    def initializeBackgroundModel(self, dataset: Dataset):
        Logger.Info('Initializing background model for dataset' + dataset.name)
        mlam = MLAttributeManager()
        self.classifier = mlam.getBackgroundClassificationModel(dataset, True)

        dba = DatasetBasedAttributes()
        self.datasetAttributes = dba.getDatasetBasedFeatures(dataset,Properties.classifier)


