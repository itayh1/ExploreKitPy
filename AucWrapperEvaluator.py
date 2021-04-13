from Logger import Logger
from Date import Date


class AucWrapperEvaluator:

     # Gets the ClassificationResults items for each of the analyzed datasets (contains the class probabilites and true
     # class for each instance)

    def  produceClassificationResults(datasets: list) -> list:
        classificationResultsPerFold = []
        for dataset in datasets:
            date = Date()
            Logger.Info("Starting to run classifier " + date.toString());
            # evaluationResults = runClassifier(properties.getProperty("classifier"), dataset.generateSet(true), dataset.generateSet(false), properties);
            # date = new Date();
            # LOGGER.info("Starting to process classification results " + date.toString());
            # ClassificationResults classificationResults = getClassificationResults(evaluationResults, dataset, properties);
            # date = new Date();
            # LOGGER.info("Done " + date.toString());
            # classificationResultsPerFold.add(classificationResults);

        return classificationResultsPerFold
