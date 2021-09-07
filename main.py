
from sklearn.model_selection import KFold, StratifiedKFold
import  pandas as pd
import arff
# from scipy.io import arff
from FilterWrapperHeuristicSearch import FilterWrapperHeuristicSearch
from Loader import Loader

def getFolds(df: pd.DataFrame, k: int) -> list:
    #TODO: make it Stratified-KFold
    cv = KFold(n_splits=k, shuffle=True, random_state=20)
    cv.get_n_splits()
    folds = []
    for train_index, test_index in cv.split(df):
        folds.append(test_index)
    return folds

def main():
    filename = '/home/itay/Documents/EKpy/ML_Background/Datasets/diabetes.arff'
    datasets = []
    classAttributeIndices = {}
    baseFolder = '/home/itay/Documents/java/ExploreKit/AutomaticFeatureGeneration-master/ML_Background/Datasets/'
    # datasets.add("/global/home/users/giladk/Datasets/heart.arff");
    # datasets.add("/global/home/users/giladk/Datasets/cancer.arff");
    # datasets.add("/global/home/users/giladk/Datasets/contraceptive.arff");
    # datasets.add("/global/home/users/giladk/Datasets/credit.arff");
    datasets.append("german_credit.arff")
    # datasets.add("/global/home/users/giladk/Datasets/diabetes.arff");
    # datasets.add("/global/home/users/giladk/Datasets/Diabetic_Retinopathy_Debrecen.arff");
    # datasets.add("/global/home/users/giladk/Datasets/horse-colic.arff");
    # datasets.add("/global/home/users/giladk/Datasets/Indian_Liver_Patient_Dataset.arff");
    # datasets.add("/global/home/users/giladk/Datasets/seismic-bumps.arff");
    # datasets.add("/global/home/users/giladk/Datasets/cardiography_new.arff");

    loader = Loader()
    randomSeed = 42
    for i in range(1):
        for datasetPath in datasets:
            if datasetPath not in classAttributeIndices.keys():
                dataset = loader.readArff(baseFolder+datasetPath, randomSeed, None, None, 0.66)
            else:
                dataset = loader.readArff(baseFolder+datasetPath, randomSeed, None, classAttributeIndices[datasetPath], 0.66)

            exp = FilterWrapperHeuristicSearch(15)
            exp.run(dataset, "_" + str(i))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
