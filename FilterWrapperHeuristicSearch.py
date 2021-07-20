
from datetime import datetime

from AucWrapperEvaluator import AucWrapperEvaluator
from Date import Date
from FilterPreRankerEvaluator import FilterPreRankerEvaluator
from OperatorsAssignmentsManager import OperatorsAssignmentsManager
from Properties import Properties
from MLFilterEvaluator import MLFilterEvaluator
from Dataset import Dataset
from Logger import Logger

class FilterWrapperHeuristicSearch:


    def __init__(self, maxIterations: int):
        self.maxIteration = maxIterations
        date = datetime.now()
        chosenOperatorAssignment = None
        topRankingAssignment = None
        evaluatedAttsCounter:int = 0

    def run(self, originalDataset: Dataset, runInfo: str):
        Logger.Info('Initializing evaluators')
        filterEvaluator = MLFilterEvaluator(originalDataset)

        preRankerEvaluator = None
        if bool(Properties.usePreRanker):
            preRankerEvaluator = FilterPreRankerEvaluator(originalDataset)

        if Properties.wrapperApproach == 'AucWrapperEvaluator':
            wrapperEvaluator = AucWrapperEvaluator()
        else:
            Logger.Error('Missing wrapper approach')
            raise Exception('Missing wrapper approach')

        experimentStartDate = Date()
        Logger.Info("Experiment Start Date/Time: " + str(experimentStartDate) + " for dataset " + originalDataset.name)

        # The first step is to evaluate the initial attributes, so we get a reference point to how well we did
        wrapperEvaluator.EvaluationAndWriteResultsToFile(originalDataset, "", 0, runInfo, True, 0, -1, -1)

        # now we create the replica of the original dataset, to which we can add columns
        dataset = originalDataset.replicateDataset()

        # Get the training set sub-folds, used to evaluate the various candidate attributes
        originalDatasetTrainingFolds = originalDataset.GenerateTrainingSetSubFolds()
        subFoldTrainingDatasets = dataset.GenerateTrainingSetSubFolds()

        date = Date()

        # We now apply the wrapper on the training subfolds in order to get the baseline score. This is the score a candidate attribute needs to "beat"
        currentScore = wrapperEvaluator.produceAverageScore(subFoldTrainingDatasets, null, null, null, null, properties);
        Logger.Info(f"Initial score: {str(currentScore)} : {date}")

        # The probabilities assigned to each instance using the ORIGINAL dataset (training folds only)
        Logger.Info(f"Producing initial classification results: {date}")
        currentClassificationProbs = wrapperEvaluator.produceClassificationResults(originalDatasetTrainingFolds)
        date = Date()
        Logger.Info(f"  .....done {date}")

        # Apply the unary operators (discretizers, normalizers) on all the original features. The attributes generated
        # here are different than the ones generated at later stages because they are included in the dataset that is
        # used to generate attributes in the iterative search phase
        Logger.Info(f"Starting to apply unary operators: {date}")
        oam = OperatorsAssignmentsManager()
        candidateAttributes = oam.applyUnaryOperators(dataset,null, filterEvaluator, subFoldTrainingDatasets, currentClassificationProbs)
        # date = new Date();
        # Logger.Info("  .....done " + date.toString());
        #
        # //Now we add the new attributes to the dataset (they are added even though they may not be included in the
        # //final dataset beacuse they are essential to the full generation of additional features
        # Logger.Info("Starting to generate and add columns to dataset:   "  + " : " + date.toString());
        # oam.GenerateAndAddColumnToDataset(dataset, candidateAttributes);
        # date = new Date();
        # Logger.Info("  .....done " + date.toString());
        #
        # //The initial dataset has been populated with the discretized/normalized features. Time to begin the search
        # int iterationsCounter = 1;
        # List<ColumnInfo> columnsAddedInthePreviousIteration = null;
        #
        # performIterativeSearch(originalDataset, runInfo,preRankerEvaluator, filterEvaluator, wrapperEvaluator, dataset, originalDatasetTrainingFolds, subFoldTrainingDatasets, currentClassificationProbs, oam, candidateAttributes, iterationsCounter, columnsAddedInthePreviousIteration);

    @staticmethod
    def getWrapper(param):
        pass