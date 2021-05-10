import errno
import os.path
import FileUtils

from ArffSaver import ArffSaver
from AttributeInfo import AttributeInfo
from Classifier import Classifier
from Column import Column
from Dataset import Dataset
from DatasetBasedAttributes import DatasetBasedAttributes
from Date import Date
from Evaluation import Evaluation
from EvaluationInfo import EvaluationInfo
from FileUtils import listFilesInDir
from Loader import Loader
from Logger import Logger
from OperatorAssignmentBasedAttributes import OperatorAssignmentBasedAttributes
from OperatorsAssignmentsManager import OperatorsAssignmentsManager
from Properties import Properties
from Serializer import Serializer

from sklearn.ensemble import RandomForestClassifier
import pandas as pd



class MLAttributeManager:

    DATASET_BASED="DatasetBased"
    OA_BASED="OperatorAssignmentBased"
    VALUES_BASED="ValuesBased"
    MERGED_ALL="merged_candidateAttributesData"
    MERGED_NO_VALUE="merged_candidateAttributes_NoValueBased"
    ITERATION = 50000

    '''
    Generates the "background" model that will be used to classify the candidate attributes of the provided dataset.
    The classification model will be generated by combining the information gathered FROM ALL OTHER DATASETS.
    '''
    def getBackgroundClassificationModel(self, dataset: Dataset, includeValueBased: bool):
        backgroundFilePath = self.getBackgroundFilePath(dataset, includeValueBased)
        path = backgroundFilePath


        # If the classification model already exists, load and return it
        if os.path.isfile(path):
            Logger.Info("Background model already exists. Extracting from " + path)
            return self.getClassificationModel(dataset, backgroundFilePath)

        #Otherwise, generate, save and return it (WARNING - takes time)
        else:
            Logger.Info("Background model doesn't exist for dataset " + dataset.name + ". Creating it...")

            # We begin by getting a list of all the datasets that need to participate in the creation of the background model
            self.generateMetaFeaturesInstances(includeValueBased)

            candidateAtrrDirectories = self.getDirectoriesInFolder(Properties.DatasetInstancesFilesLocation)
            self.generateBackgroundARFFFileForDataset(dataset, backgroundFilePath, candidateAtrrDirectories, includeValueBased)

            # now we load the contents of the ARFF file into an Instances object and train the classifier
            data = self.getInstancesFromARFF(backgroundFilePath)
            return self.buildClassifierModel(backgroundFilePath, data)

    def getInstancesFromARFF(self, backgroundFilePath: str):
        # BufferedReader reader = new BufferedReader(new FileReader(backgroundFilePath + ".arff"));
        data = Loader().readArffAsDataframe(backgroundFilePath + '.arff')
        Logger.Info('reading from file ' + backgroundFilePath + '.arff')
        # ArffLoader.ArffReader arffReader = new ArffLoader.ArffReader(reader);
        # Instances data = arffReader.getData();
        # data.setClassIndex(data.numAttributes() - 1);
        return data

    def buildClassifierModel(self, backgroundFilePath: str, data):
        # the chosen classifier
        classifier = RandomForestClassifier()
        # classifier.setNumExecutionSlots(Integer.parseInt(properties.getProperty("numOfThreads")));

        # classifier.buildClassifier(data);
        classifier.fit(data.drop(['class']), data['class'])
        file = backgroundFilePath + '.arff'
        FileUtils.deleteFile(file)

        Logger.Info('Saving classifier model ' + backgroundFilePath)
        self.writeClassifierTobackgroundFile(backgroundFilePath, classifier)
        return classifier

    def writeClassifierTobackgroundFile(self, backgroundFilePath: str, classifier):
        # now we write the classifier to file prior to returning the object
        Serializer.Serialize(backgroundFilePath, classifier)

    def generateBackgroundARFFFileForDataset(self, dataset:Dataset, backgroundFilePath: str, candidateAttrDirectories: list, includeValueBased: bool):
        addHeader = True
        for candidateAttrDirectory in candidateAttrDirectories:

            if (not candidateAttrDirectory.__contains__(dataset.name)) and FileUtils.listFilesInDir(candidateAttrDirectory)!=None: #none means dir exist

                merged = self.getMergedFile(candidateAttrDirectory,includeValueBased)
                if merged is not None:
                    MLAttributeManager.addArffFileContentToTargetFile(backgroundFilePath, merged[0].getAbsolutePath(),addHeader)
                    addHeader = False

                else:
                    instances = [] #List<Instances> instances = new ArrayList<>();
                    for file in listFilesInDir(candidateAttrDirectory):
                        if (file.contains('.arff') and not(not includeValueBased and file.contains(self.VALUES_BASED)) and not(file.contains('merged'))):
                            absFilePath = os.path.abspath(file)
                            instance = Loader().readArffAsDataframe(absFilePath)
                            instances.append(instance)

                        else:
                            Logger.Info(f'Skipping file: {file}')

                    mergedFile = self.mergeInstancesToFile(includeValueBased, candidateAttrDirectory, instances)
                    if mergedFile is None:
                        continue
                    self.addArffFileContentToTargetFile(backgroundFilePath, FileUtils.getAbsPath(mergedFile), addHeader)
                    addHeader = False

    def mergeInstancesToFile(self, includeValueBased: bool, directory: str, instances: list):
        if len(instances) == 0:
            return None

        toMerge = instances[0]
        for i in range(1, len(instances)):
            toMerge = self.mergeDataframes(toMerge, instances[i]) #Instances.mergeInstances(toMerge, instances.get(i));

        saver = ArffSaver()
        saver.setInstances(toMerge)

        if includeValueBased:
            mergedFile = os.path.join(directory, self.MERGED_ALL + '.arff')
        else:
            mergedFile = os.path.join(directory, self.MERGED_NO_VALUE + '.arff')

        saver.setFile(mergedFile)
        saver.writeBatch()
        return mergedFile

    # TODO: replace 'mergeInstances' built-in function
    def mergeDataframes(self, first, second):
        return pd.concat([first, second], axis=1)

    def getMergedFile(self, directory: str, includeValueBased:bool):
        if includeValueBased:
            merged = filter(lambda name: name.contains(self.MERGED_ALL), listFilesInDir(directory))
            merged = list(merged)
            if len(merged) == 1:
                return merged

        if not includeValueBased:
            filter(lambda name: name.contains(self.MERGED_NO_VALUE), listFilesInDir(directory))
            if len(merged) == 1:
                return merged

        return None

    def generateMetaFeaturesInstances(self, includeValueBased: bool):
        datasetFilesForBackgroundArray = self.getOriginalBackgroundDatasets()
        for datasetForBackgroundModel in datasetFilesForBackgroundArray:
            possibleFolderName = Properties.DatasetInstancesFilesLocation + \
                                 FileUtils.getFilenameFromPath(datasetForBackgroundModel) + '_' + str(Properties.randomSeed)

            if not os.path.isdir(possibleFolderName):
                loader = Loader()
                Logger.Info("Getting candidate attributes for " + datasetForBackgroundModel)
                backgroundDataset = loader.readArff(datasetForBackgroundModel, int(Properties.randomSeed), None, None, 0.66)
                self.createDatasetMetaFeaturesInstances(backgroundDataset, includeValueBased)

    def getDirectoriesInFolder(self, folder:str):
        for (_, directories, _) in os.walk(folder):
            break
        if (directories is None) or (len(directories) == 0):
            raise Exception('getBackgroundClassificationModel -> no directories for meta feature instances')
        return directories

    # Receives a dataset object and generates an Instance object that contains a set of attributes for EACH candidate attribute
    def createDatasetMetaFeaturesInstances(self, dataset: Dataset, includeValueBased: bool):
        directoryForDataset = Properties.DatasetInstancesFilesLocation + dataset.name
        # File[] files;

        if os.path.isdir(directoryForDataset):
            _, _, filenames = next(os.walk(directoryForDataset))
            if (filenames is not None) and (len(filenames)!=0):
                Logger.Info('Candidate attributes for ' + dataset.name + ' were already calculated')
                return

        try:
            os.mkdir(directoryForDataset)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                Logger.Warn(f'getDatasetMetaFeaturesInstances -> Error creating directory {directoryForDataset}\nError: {ex}')
                raise

        # List<String> metadataTypes;
        if includeValueBased:
            # This is the line that activates the (time consuming) background datasets feature generation process
            self.generateTrainingSetDatasetAttributes(dataset)
            metadataTypes = [self.DATASET_BASED, self.OA_BASED, self.VALUES_BASED]
        else:
            # for pre-ranker model
            self.generateTrainingSetDatasetAttributesWithoutValues(dataset)
            metadataTypes = [self.ATASET_BASED, self.OA_BASED]

        self.appendARFFFilesPerMetadataTypeForDataset(directoryForDataset, metadataTypes)

    def appendARFFFilesPerMetadataTypeForDataset(self, directoryForDataset: str, metaTypes: list):
        classifiers = Properties.classifiersForMLAttributesGeneration.split(',')
        seperator = os.path.sep
        for classifier in classifiers:
            for metadataType in metaTypes:
                i=1
                fileName = directoryForDataset + seperator + classifier + "_" + metadataType + "_candidateAttributesData" + i + '.arff'
                targetFile = directoryForDataset + seperator + classifier + "_" + metadataType + "_candidateAttributesData" + 0
                arffToAppendFrom = fileName
                while os.path.exists(arffToAppendFrom):
                    MLAttributeManager.addArffFileContentToTargetFile(targetFile, fileName, False)
                    FileUtils.deleteFile(arffToAppendFrom)
                    i += 1
                    fileName = directoryForDataset + seperator + classifier + "_" + metadataType + "_candidateAttributesData" + i + '.arff'
                    arffToAppendFrom = fileName

                mainFile = targetFile + '.arff'
                FileUtils.renameFile(mainFile, directoryForDataset + seperator + classifier + "_" + metadataType + "_candidateAttributesData" +  '.arff')

    def getBackgroundFilePath(self, dataset: Dataset, includeValueBased: bool):
        backgroundFilePath = ''
        if includeValueBased:
             backgroundFilePath = Properties.backgroundClassifierLocation + 'background_' + dataset.name + '_' + 'DatasetBased_OperatorAssignmentBased_ValuesBased' + '_classifier_obj'
        else:
             backgroundFilePath = Properties.backgroundClassifierLocation + 'background_' + dataset.name + '_' + 'DatasetBased_OperatorAssignmentBased' + '_classifier_obj'
        return backgroundFilePath

    def getClassificationModel(self, dataset: Dataset, backgroundFilePath: str):
        try:
            backgroundModel = Serializer.Deserialize(backgroundFilePath)
            return backgroundModel

        except Exception as ex:
            Logger.Error("getBackgroundClassificationModel -> Error reading classifier for dataset " + dataset.getName()
                         + "  from file: " + backgroundFilePath + "  :  " + ex)
            return None

    # return absolute path of the dataset
    def getOriginalBackgroundDatasets(self):
        datasetFolder = Properties.originalBackgroundDatasetsLocation
        datasetsFilesList = [os.path.join(datasetFolder, f) for f in os.listdir(datasetFolder) if os.path.isfile(os.path.join(datasetFolder, f))]
        if len(datasetsFilesList)==0:
            raise Exception('generateMetaFeaturesInstances -> no files in ' + datasetFolder)
        return datasetsFilesList

     # Generates the candidate attributes and then generates the meta-features for each of them. This is time consuming because you need to generate
     # all the values of each candidate features, then calculate the meta-features. Please note that the label (i.e. is the candidate attribute good or
     # bad) is dependent on the improvement obtained by re-training the model and evaluating it.
    def generateTrainingSetDatasetAttributes(self, dataset):
        Logger.Info("Generating dataset attributes for dataset: " + dataset.name)

        # DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        startDate = Date()
        # The structure: Classifier -> candidate feature (operator assignment, to be exact) -> meta-feature type -> A map of feature indices and values
        # { classifier:
        #     { OperatorAssigment:
        #           { meta-feature type: {indice, value}}
        # TreeMap<String, HashMap<OperatorAssignment,HashMap<String,TreeMap<Integer,AttributeInfo>>>> candidateAttributesList = new TreeMap<>()
        candidateAttributesList = {}

        classifiers = Properties.classifiersForMLAttributesGeneration.split(',')

        # obtaining the attributes for the dataset itself is straightforward
        dba = DatasetBasedAttributes()
        for classifier in classifiers:
            candidateAttributesList[classifier] = {}
            originalAuc = self.getOriginalAuc(dataset, classifier)

            # Generate the dataset attributes
            datasetAttributes = dba.getDatasetBasedFeatures(dataset, classifier)

            # now we need to generate the candidate attributes and evaluate them. This requires a few preliminary steps:
            # 1) Replicate the dataset and create the discretized features and add them to the dataset
            unaryOperators = OperatorsAssignmentsManager.getUnaryOperatorsList()

            # The unary operators need to be evaluated like all other operator assignments (i.e. attribtues generation)
            unaryOperatorAssignments = OperatorsAssignmentsManager.getOperatorAssignments(dataset, None, unaryOperators, int(Properties.maxNumOfAttsInOperatorSource))
            replicatedDataset = self.generateDatasetReplicaWithDiscretizedAttributes(dataset, unaryOperatorAssignments)

            # 2) Obtain all other operator assignments (non-unary). IMPORTANT: this is applied on the REPLICATED dataset so we can take advantage of the discretized features
            nonUnaryOperators = OperatorsAssignmentsManager.getNonUnaryOperatorsList()
            nonUnaryOperatorAssignments = OperatorsAssignmentsManager.getOperatorAssignments(replicatedDataset, None, nonUnaryOperators, int(Properties.maxNumOfAttsInOperatorSource))

            # 3) Generate the candidate attribute and generate its attributes
            nonUnaryOperatorAssignments.addAll(unaryOperatorAssignments);

            # oaList.parallelStream().forEach(oa -> {
            # ReentrantLock wrapperResultsLock = new ReentrantLock();
            # for (OperatorAssignment oa : nonUnaryOperatorAssignments) {
            position = [0] #new int[]{0};

            # TODO: keep it pararell, temporary changed to single thread
            # nonUnaryOperatorAssignments.parallelStream().forEach(oa -> {
            for oa in nonUnaryOperatorAssignments:
                try:
                    datasetReplica = dataset.replicateDataset()

                    # Here we generate all the meta-features that are "parent dependent" and do not require us to generate the values of the new attribute
                    oaba = OperatorAssignmentBasedAttributes()

                    # TreeMap < Integer, AttributeInfo >
                    candidateAttributeValuesFreeMetaFeatures = oaba.getOperatorAssignmentBasedMetaFeatures(dataset, oa)

                    # ColumnInfo candidateAttribute;
                    try:
                        candidateAttribute = OperatorsAssignmentsManager.generateColumn(datasetReplica, oa, True)
                    except:
                        candidateAttribute = OperatorsAssignmentsManager.generateColumn(datasetReplica, oa, True)

                    datasetReplica.addColumn(candidateAttribute)

                    evaluationInfo = self.runClassifier(classifier, datasetReplica.generateSet(True), datasetReplica.generateSet(False))
                    evaluationResults1 = evaluationInfo.getEvaluationStats()

                    # synchronized (this){ #TODO: part of the pararell stream
                    #     candidateAttributesList.get(classifier).put(oa, new HashMap<>());
                    #     candidateAttributesList.get(classifier).get(oa).put(DATASET_BASED, datasetAttributes);
                    candidateAttributesList[classifier][oa][MLAttributeManager.DATASET_BASED] =  datasetAttributes
                    # Add the identifier of the classifier that was used
                    classifierAttribute = AttributeInfo("Classifier", Column.columnType.Discrete, self.getClassifierIndex(classifier), 3)
                    candidateAttributeValuesFreeMetaFeatures[candidateAttributeValuesFreeMetaFeatures.size()] = classifierAttribute
                    candidateAttributesList[classifier][oa][MLAttributeManager.OA_BASED] = candidateAttributeValuesFreeMetaFeatures

                    candidateAttributeValuesDependentMetaFeatures = oaba.getGeneratedAttributeValuesMetaFeatures(dataset, oa, candidateAttribute)
                    candidateAttributesList[classifier][oa][MLAttributeManager.VALUES_BASED] = candidateAttributeValuesDependentMetaFeatures
                    candidateAttributesList[classifier][oa][MLAttributeManager.OA_BASED][candidateAttributesList[classifier][oa][MLAttributeManager.OA_BASED].size()] = self.createClassAttribute(originalAuc, datasetReplica, evaluationResults1)


                    # wrapperResultsLock.lock(); #TODO: part of the pararell stream
                    if (len(candidateAttributesList[classifier]) % 1000) == 0:
                        date = Date()
                        Logger.Info(date.__str__() + ": Finished processing " + ((position[0] * MLAttributeManager.ITERATION) + len(candidateAttributesList[classifier]) + '/' + nonUnaryOperatorAssignments.size() + ' elements for background model'))

                    if (len(candidateAttributesList[classifier]) % MLAttributeManager.ITERATION) == 0:
                        self.savePartArffCandidateAttributes(candidateAttributesList,classifier,dataset,position[0])
                        position[0] += 1
                        candidateAttributesList[classifier].clear()
                    # wrapperResultsLock.unlock(); #TODO: part of the pararell stream
                except Exception as ex:
                    Logger.Error("Error in ML features generation : " + oa.getName() + "  :  " + ex)


            self.savePartArffCandidateAttributes(candidateAttributesList,classifier,dataset,position[0])

        finishDate = Date()
        diffInMillies = finishDate - startDate
        Logger.Info("Getting candidate attributes for dataset " + dataset.getName() + " took " + diffInMillies.second + " seconds")

    def generateTrainingSetDatasetAttributesWithoutValues(self, dataset):
        Logger.Info("Generating dataset attributes for dataset: " + dataset.name)

        # DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        startDate = Date()
        # The structure: Classifier -> candidate feature (operator assignment, to be exact) -> meta-feature type -> A map of feature indices and values
        # { classifier:
        #     { OperatorAssigment:
        #           { meta-feature type: {indice, value}}
        # TreeMap<String, HashMap<OperatorAssignment,HashMap<String,TreeMap<Integer,AttributeInfo>>>> candidateAttributesList = new TreeMap<>()
        candidateAttributesList = {}

        classifiers = Properties.classifiersForMLAttributesGeneration.split(',')

        # obtaining the attributes for the dataset itself is straightforward
        dba = DatasetBasedAttributes()
        for classifier in classifiers:
            candidateAttributesList[classifier] = {}
            originalAuc = self.getOriginalAuc(dataset, classifier)

            # Generate the dataset attributes
            datasetAttributes = dba.getDatasetBasedFeatures(dataset, classifier)

            # now we need to generate the candidate attributes and evaluate them. This requires a few preliminary steps:
            # 1) Replicate the dataset and create the discretized features and add them to the dataset
            unaryOperators = OperatorsAssignmentsManager.getUnaryOperatorsList()

            # The unary operators need to be evaluated like all other operator assignments (i.e. attribtues generation)
            unaryOperatorAssignments = OperatorsAssignmentsManager.getOperatorAssignments(dataset, None, unaryOperators, int(Properties.maxNumOfAttsInOperatorSource))
            replicatedDataset = self.generateDatasetReplicaWithDiscretizedAttributes(dataset, unaryOperatorAssignments)

            # 2) Obtain all other operator assignments (non-unary). IMPORTANT: this is applied on the REPLICATED dataset so we can take advantage of the discretized features
            nonUnaryOperators = OperatorsAssignmentsManager.getNonUnaryOperatorsList()
            nonUnaryOperatorAssignments = OperatorsAssignmentsManager.getOperatorAssignments(replicatedDataset, None, nonUnaryOperators, int(Properties.maxNumOfAttsInOperatorSource))

            # 3) Generate the candidate attribute and generate its attributes
            nonUnaryOperatorAssignments.addAll(unaryOperatorAssignments);

            # oaList.parallelStream().forEach(oa -> {
            # ReentrantLock wrapperResultsLock = new ReentrantLock();
            # for (OperatorAssignment oa : nonUnaryOperatorAssignments) {
            position = [0] #new int[]{0};

            # TODO: keep it pararell, temporary changed to single thread
            # nonUnaryOperatorAssignments.parallelStream().forEach(oa -> {
            for oa in nonUnaryOperatorAssignments:
                try:
                    datasetReplica = dataset.replicateDataset()

                    # Here we generate all the meta-features that are "parent dependent" and do not require us to generate the values of the new attribute
                    oaba = OperatorAssignmentBasedAttributes()

                    # TreeMap < Integer, AttributeInfo >
                    candidateAttributeValuesFreeMetaFeatures = oaba.getOperatorAssignmentBasedMetaFeatures(dataset, oa)


                    evaluationInfo = self.runClassifier(classifier, datasetReplica.generateSet(True), datasetReplica.generateSet(False))
                    evaluationResults1 = evaluationInfo.getEvaluationStats()

                    # synchronized (this){ #TODO: part of the pararell stream
                    #     candidateAttributesList.get(classifier).put(oa, new HashMap<>());
                    #     candidateAttributesList.get(classifier).get(oa).put(DATASET_BASED, datasetAttributes);
                    candidateAttributesList[classifier][oa][MLAttributeManager.DATASET_BASED] =  datasetAttributes
                    # Add the identifier of the classifier that was used
                    classifierAttribute = AttributeInfo("Classifier", Column.columnType.Discrete, self.getClassifierIndex(classifier), 3)
                    candidateAttributeValuesFreeMetaFeatures[candidateAttributeValuesFreeMetaFeatures.size()] = classifierAttribute
                    candidateAttributesList[classifier][oa][MLAttributeManager.OA_BASED] = candidateAttributeValuesFreeMetaFeatures

                    # candidateAttributeValuesDependentMetaFeatures = oaba.getGeneratedAttributeValuesMetaFeatures(dataset, oa, candidateAttribute)
                    # candidateAttributesList[classifier][oa][MLAttributeManager.VALUES_BASED] = candidateAttributeValuesDependentMetaFeatures
                    candidateAttributesList[classifier][oa][MLAttributeManager.OA_BASED][candidateAttributesList[classifier][oa][MLAttributeManager.OA_BASED].size()] = self.createClassAttribute(originalAuc, datasetReplica, evaluationResults1)


                    # wrapperResultsLock.lock(); #TODO: part of the pararell stream
                    if (len(candidateAttributesList[classifier]) % 1000) == 0:
                        date = Date()
                        Logger.Info(date.__str__() + ": Finished processing " + ((position[0] * MLAttributeManager.ITERATION) + len(candidateAttributesList[classifier]) + '/' + nonUnaryOperatorAssignments.size() + ' elements for background model'))

                    if (len(candidateAttributesList[classifier]) % MLAttributeManager.ITERATION) == 0:
                        self.savePartArffCandidateAttributes(candidateAttributesList,classifier,dataset,position[0])
                        position[0] += 1
                        candidateAttributesList[classifier].clear()
                    # wrapperResultsLock.unlock(); #TODO: part of the pararell stream
                except Exception as ex:
                    Logger.Error("Error in ML features generation : " + oa.getName() + "  :  " + ex)


            self.savePartArffCandidateAttributes(candidateAttributesList,classifier,dataset,position[0])

        finishDate = Date()
        diffInMillies = finishDate - startDate
        Logger.Info("Getting candidate attributes for dataset " + dataset.getName() + " took " + diffInMillies.second + " seconds")


    # @param originalAuc to calculate the deltaauc
    # @param datasetReplica
    # @param evaluationResults1
    # @return AttributeInfo of class attribute
    def createClassAttribute(self, originalAuc: float, datasetReplica: Dataset, evaluationResults1:Evaluation):
        auc = self.CalculateAUC(evaluationResults1, datasetReplica)
        deltaAuc = auc - originalAuc;
        if deltaAuc > 0.01:
            classAttribute =  AttributeInfo("classAttribute", Column.columnType.Discrete, 1, 2)
            Logger.Info("found positive match with delta " + deltaAuc)
        else:
            classAttribute = AttributeInfo("classAttribute", Column.columnType.Discrete, 0, 2)
        return classAttribute



    def runClassifier(self, classifierName: str, trainingSet, testSet):
            try:
                classifier = Classifier(classifierName)
                classifier.buildClassifier(trainingSet)
                classifier.evaluateClassifier(testSet)
                # The overall classification statistics
                # Evaluation evaluation;
                evaluation = Evaluation(trainingSet)
                evaluation.evaluateModel(classifier, testSet)

                # The confidence score for each particular instance
                scoresDist = []
                for i in range(testSet.size()):
                    testInstance = testSet[i]
                    score = classifier.distributionForInstance(testInstance)
                    scoresDist.append(score)

                return EvaluationInfo(evaluation, scoresDist)

            except Exception as ex:
                Logger.Error("problem running classifier. Exception:"+ex)

            return None

    # Replcates the provided dataset and created discretized versions of all relevant columns and adds them to the datast
    def generateDatasetReplicaWithDiscretizedAttributes(self, dataset: Dataset, unaryOperatorAssignments: list):
            replicatedDatast = dataset.replicateDataset()
            for oa in unaryOperatorAssignments:
                ci = OperatorsAssignmentsManager.generateColumn(replicatedDatast, oa, True)
                replicatedDatast.addColumn(ci)
            return replicatedDatast


    def getOriginalAuc(self, dataset: Dataset, classifier: str):
            # For each dataset and classifier combination, we need to get the results on the "original" dataset so we can later compare
            originalRunEvaluationInfo = self.runClassifier(classifier, dataset.generateSet(True), dataset.generateSet(False))
            originalRunEvaluationResults = originalRunEvaluationInfo.getEvaluationStats()
            return self.CalculateAUC(originalRunEvaluationResults, dataset)

    # Returns an integer that is used to represent the classifier in the generated Instances object
    def getClassifierIndex(classifier: str) -> int:
        if classifier == 'J48':
            return 0
        elif classifier == "SVM":
            return 1
        elif classifier == "RandomForest":
            return 2
        else:
            raise Exception("Unidentified classifier")

    def savePartArffCandidateAttributes(self,candidateAttributesList: dict, classifier:str, dataset: Dataset, position: int):
        if len(candidateAttributesList[classifier]) > 0:
            filePath = Properties.DatasetInstancesFilesLocation + dataset.name
            self.saveInstancesForDatasetAttributes(filePath, candidateAttributesList, str(position))

    def saveInstancesForDatasetAttributes(self, directoryToSaveARFF:str, datasetAttributeValues: dict, part: str):
        datasetInstances = self.generateValuesMatrices(datasetAttributeValues)
        self.saveSerAndArffFiles(directoryToSaveARFF, datasetInstances, part)

    # creates and saves arff files (1 file per classifier and metadata type)
    # @param directoryForDataset directory to save the arff and ser files that will be created
    # @param datasetInstances Instances to save in files
    # @param part
    def saveSerAndArffFiles(self, directoryForDataset, datasetInstances: dict, part: str):
        # for (Map.Entry<String, HashMap<String, Instances>> classifierInstances : datasetInstances.entrySet()){
        for classifierKey, classifierValue in datasetInstances.items():
            # for (Map.Entry<String, Instances> instancesByMetadataType : classifierInstances.getValue().entrySet()){
            for instancesByMetadataTypeKey, instancesByMetadataTypeValue in classifierValue.items():
                fileName = classifierKey + "_" + instancesByMetadataTypeKey + "_candidateAttributesData" + part + ".ser"
        #         FileOutputStream fout = new FileOutputStream(directoryForDataset.toString() + File.separator +  fileName, true);
        #         ObjectOutputStream oos = new ObjectOutputStream(fout);
        #         oos.writeObject(instancesByMetadataType.getValue());
        #         oos.close();
        #         fout.close();
        #
        #         ArffSaver saver = new ArffSaver();
        #         saver.setInstances(instancesByMetadataType.getValue());
        #         saver.setFile(new File(directoryForDataset.toString() + File.separator + classifierInstances.getKey() + "_" + instancesByMetadataType.getKey()  + "_candidateAttributesData" + part + ".arff"));
        #         saver.writeBatch();
        #
        #         //now delete the .ser file
        #         File fileToDelete = new File(directoryForDataset.toString() + File.separator + fileName);
        #         if (!fileToDelete.delete()) {
        #             LOGGER.warn("saveSerAndArffFiles -> Failed to delete file: " + fileToDelete.getAbsolutePath());
        #         }
        #     }
        # }

    def CalculateAUC(self, evaluation: Evaluation, dataset: Dataset) -> float:
        auc = 0.0
        for i in range(dataset.getNumOfClasses()):
            auc += evaluation.areaUnderROC(i)

        auc = auc / dataset.getNumOfClasses()
        return auc


    # Creates  data matrices from a HashMap of AttributeInfo objects per metadata type and per classifier
    # @param datasetAttributeValues
    # @return TreeMap of classifiers -> HashMap of type of metadata of Instances
    def generateValuesMatrices(self, datasetAttributeValues: dict) -> dict:
        instancesMapPerClassifier = {}
        # for(Map.Entry<String, HashMap<OperatorAssignment, HashMap<String, TreeMap<Integer, AttributeInfo>>>> entry : datasetAttributeValues.entrySet()) {
        for attrKey, attrValue in datasetAttributeValues.items():
            instancesMapPerClassifier[attrKey] = self.generateValueMatrixPerClassifier(attrValue)

        return instancesMapPerClassifier

    # Appends the content of an ARFF file to the end of another. Used to generate the all-but-one ARFF files
    #  * @param targetFilePath The path of the file to which we want to write
    #  * @param arffFilePath the source file
    #  * @param addHeader whether the ARFF header needs to be copied as well. When this is set to 'true' the file will be overwritten
    @staticmethod
    def addArffFileContentToTargetFile(targetFilePath: str, arffFilePath: str, addHeader: bool):
        targetFilePath += ".arff"
        fileReader =  open(arffFilePath, 'r')
        if addHeader:
            fileWriter = open(targetFilePath, 'w')
        else:
            fileWriter = open(targetFilePath, 'a')

        foundData = False
        line = fileReader.readline()

        while line:
            if addHeader:
                fileWriter.write(line + '\n')
            else:
                if not foundData and (line.lower().find("@data") != -1):
                    foundData = True
                    line = fileReader.readline()
                if foundData:
                    fileWriter.write(line + '\n')

        fileReader.close()
        fileWriter.flush()
        fileWriter.close()


