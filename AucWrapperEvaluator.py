import numpy as np
from sklearn.metrics import roc_auc_score

from ClassificationItem import ClassificationItem
from ClassificationResults import ClassificationResults
from Classifier import Classifier
from Dataset import Dataset
from EvaluationInfo import EvaluationInfo
from Logger import Logger
from Date import Date
from Properties import Properties


class AucWrapperEvaluator:

     # Gets the ClassificationResults items for each of the analyzed datasets (contains the class probabilites and true
     # class for each instance)

    def produceClassificationResults(self, datasets: list) -> list:
        classificationResultsPerFold = []
        for dataset in datasets:
            date = Date()
            Logger.Info("Starting to run classifier " + str(date))
            trainSet = dataset.generateSet(True)
            testSet = dataset.generateSet(False)
            evaluationResults = self.runClassifier(Properties.classifier, trainSet, testSet)
            date = Date()
            Logger.Info("Starting to process classification results " + str(date))
            classificationResults = self.getClassificationResults(evaluationResults, dataset, testSet)
            date = Date()
            Logger.Info("Done " + str(date))
            classificationResultsPerFold.append(classificationResults)

        return classificationResultsPerFold

    def runClassifier(self, classifierName: str, trainingSet, testSet) -> EvaluationInfo:
        try:
            classifier = Classifier(classifierName)
            classifier.buildClassifier(trainingSet)

            # evaluation = new Evaluation(trainingSet);
            # evaluation.evaluateModel(classifier, testSet)
            evaluationInfo = classifier.evaluateClassifier(testSet)

            return evaluationInfo

        except Exception as ex:
            Logger.Error("problem running classifier " + str(ex))

        return None

    # Obtains the classification probabilities assigned to each instance and returns them as a ClassificationResults object
    def getClassificationResults(self, evaluation, dataset: Dataset, testSet):
        date = Date()

        # used for validation - by making sure that that the true classes of the instances match we avoid "mix ups"
        actualTargetColumn = dataset.df[dataset.targetClass]

        classificationItems = []
        counter = 0
        actualValues = testSet[dataset.targetClass].values
        predDistribution = evaluation.getScoreDistribution()
        for i in range(predDistribution.shape[0]):
            # if ((counter%10000) == 0) {
            #     if ((int) prediction.actual() != (Integer) actualTargetColumn.getValue(dataset.getIndicesOfTestInstances().get(counter))) {
            #         if (dataset.getTestDataMatrixWithDistinctVals() == null || dataset.getTestDataMatrixWithDistinctVals().length == 0) {
            #             throw new Exception("the target class values do not match");
            #         }
            #     }
            # }
            counter += 1
            ci = ClassificationItem(actualValues[i],predDistribution[i])
            classificationItems.append(ci)

        # Now generate all the statistics we may want to use
        auc = self.CalculateAUC(evaluation, dataset, testSet)

        logloss = self.CalculateLogLoss(evaluation, dataset)

        # We calcualte the TPR/FPR rate. We do it ourselves because we want all the values
        tprFprValues = self.calculateTprFprRate(evaluation, dataset, testSet)

        # The TRR/FPR values enable us to calculate the precision/recall values.
        recallPrecisionValues = self.calculateRecallPrecisionValues(dataset, tprFprValues,
                float(Properties.precisionRecallIntervals))

        # Next, we calculate the F-Measure at the selected points
        fMeasureValuesPerRecall = {}
        fMeasurePrecisionValues = Properties.FMeausrePoints
        for recallVal in fMeasurePrecisionValues:
            recall = float(recallVal)
            precision = recallPrecisionValues[recall]
            F1Measure = (2*precision*recall)/(precision+recall)
            fMeasureValuesPerRecall[recall] = F1Measure

        classificationResults = ClassificationResults(classificationItems, auc, logloss, tprFprValues, recallPrecisionValues, fMeasureValuesPerRecall)

        return classificationResults


    def CalculateAUC(self, evaluation, dataset: Dataset, testSet) -> float:
        return roc_auc_score(testSet[dataset.targetClass], evaluation.scoreDistPerInstance[:, 1])

    def CalculateLogLoss(self, evaluation, dataset):
        probs = evaluation.getScoreDistribution()
        probs = np.max(probs, axis=1)
        probs = np.maximum(np.minimum(probs, 1 - 1E-15), 1E-15)
        probs = np.log(probs)
        logLoss = np.sum(probs) / probs.shape[0]
        return logLoss

    def getClassificationItemList(self, testSet, evaluation):
         assert testSet.shape[0] == evaluation.getScoreDistribution().shape[0]
         # classificationItems = []
         probs = evaluation.getScoreDistribution()
         classes = evaluation.getEvaluationStats().classes_
         classificationItems = [ClassificationItem(classIndex, dict(zip(classes, prob))) for classIndex, prob in zip(testSet['class'].values, probs)]
         # for i in range(testSet.shape[0]):
         #     classificationItems.append(ClassificationItem(testSet.iloc[i], dict(zip(classes, probs[i]))))
         return classificationItems

    # Used to calculate all the TPR-FPR values of the provided evaluation
    def calculateTprFprRate(self, evaluation, dataset, testSet) -> dict:
        date = Date()
        Logger.Info("Starting TPR/FPR calculations : " + str(date))

        # trpFprRates = {}

        # we convert the results into a format that's more comfortable to work with
        classificationItems = self.getClassificationItemList(testSet, evaluation)
        # for (Prediction prediction: evaluation.predictions()) {
        #     ClassificationItem ci = new ClassificationItem((int)prediction.actual(),((NominalPrediction)prediction).distribution());
        #     classificationItems.add(ci);
        # }

        # now we need to know what is the minority class and the number of samples for each class
        minorityClassIndex = dataset.getMinorityClassIndex()
        numOfNonMinorityClassItems = 0 #all non-minority class samples are counted together (multi-class cases)
        for cls in dataset.getNumOfRowsPerClassInTestSet().keys():
            if cls != minorityClassIndex:
                numOfNonMinorityClassItems += dataset.getNumOfRowsPerClassInTestSet()[cls]

        # sort all samples by their probability of belonging to the minority class
        classificationItems.sort(reverse=True, key=lambda x:x.getProbabilitiesOfClass(minorityClassIndex))
        # Collections.sort(classificationItems, new ClassificationItemsComparator(minorityClassIndex));
        # Collections.reverse(classificationItems);

        tprFprValues = {}
        tprFprValues[0.0] = 0.0
        minoritySamplesCounter = 0
        majoritySamplesCounter = 0
        currentProb = 2
        for ci in classificationItems:
            currentSampleProb = ci.getProbabilitiesOfClass(minorityClassIndex)
            # if the probability is different, time to update the TPR/FPR statistics
            if currentSampleProb != currentProb:
                tpr =  minoritySamplesCounter/dataset.getNumOfRowsPerClassInTestSet()[minorityClassIndex]
                fpr = majoritySamplesCounter/numOfNonMinorityClassItems
                tprFprValues[tpr] = fpr
                currentProb = currentSampleProb

            if ci.getTrueClass() == minorityClassIndex:
                minoritySamplesCounter += 1
            else:
                majoritySamplesCounter += 1

        tprFprValues[1.0] = 1.0
        tprFprValues[1.0001] = 1.0
        date = Date()
        Logger.Info("Done : " + str(date))
        return tprFprValues

    # Used to calculate the recall/precision values from the TPR/FPR values. We use the recall values as the basis for
    # our calculation because they are monotonic and becuase it enables the averaging of different fold values
    def calculateRecallPrecisionValues(self, dataset: Dataset, tprFprValues: dict, recallInterval: float):
        # start by getting the number of samples in the minority class and in other classes
        minorityClassIndex = dataset.getMinorityClassIndex()
        numOfMinorityClassItems = dataset.getNumOfRowsPerClassInTestSet()[minorityClassIndex]
        numOfNonMinorityClassItems = 0 # all non-minority class samples are counted together (multi-class cases)
        for idx, value in dataset.getNumOfRowsPerClassInTestSet().items():
            if idx != minorityClassIndex:
                numOfNonMinorityClassItems += value

        recallPrecisionValues = {}
        for i in np.arange(0, 1+1e-5, recallInterval):
            recallKey = self.getClosestRecallValue(tprFprValues, i)  # the recall is the TPR
            #TODO: ask Gilad about recallKey and tprFprValue of 0.0
            try:
                precision = (recallKey*numOfMinorityClassItems)/((recallKey*numOfMinorityClassItems) + (tprFprValues[recallKey]*numOfNonMinorityClassItems))
            except ZeroDivisionError:
                precision = 0
            # if np.isnan(precision):
            #     precision = 0
            recallPrecisionValues[round(i, 2)] = precision

        return recallPrecisionValues


    # Returns the ACTUAL recall value that is closest to the requested value. It is important to note that there are
    # no limitations in this function, so in end-cases the function may return strange results.
    def getClosestRecallValue(self, tprFprValues: dict, recallVal: float) -> float:
        for key in tprFprValues.keys():
            if key >= recallVal:
                return key
        return 0

