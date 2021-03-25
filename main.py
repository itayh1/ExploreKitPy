
from sklearn.model_selection import KFold
import  pandas as pd
import arff
# from scipy.io import arff
from Loader import Loader

def getFolds(df: pd.DataFrame, k: int) -> list:
    cv = KFold(n_splits=k, shuffle=True, random_state=20)
    cv.get_n_splits()
    folds = []
    for train_index, test_index in cv.split(df):
        folds.append(test_index)
    return folds

def main():
    filename = '/home/itay/Documents/EKpy/ML_Background/Datasets/diabetes.arff'
    # randomSeed = 42
    # loader = Loader()
    # dataset = loader.readArff(filename, randomSeed, None, None, 0.66)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
