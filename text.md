EKpy:
files in DatasetsForMetaModel are the arrfs that we train upon and create 
from them the candidate features for training.

we go for each dataset and calculate its features: Dataset based, Value based and OperationAssignment based.
Dataset based: the meta-features for each candidate is the same for all rows in dataset.
Value based:
OperationAssignment based:

`MLFilterEvaluator` 

`MLAttributesManager` creates the meta-classifier for each dataset.
for each dataset:
1. Get DatasetBased attributes using `DatasetBasedAttributes`.
2. Generate candidate features
   1. Discretize features and add to Dataset (apply all unary operators)
   2. Apply all non-unary operators using `OperationAssignmentManager`, 
   as Operation Assignments (candidate features that haven't evaluated yet/generated values).
   3. Generate all the meta-features that are Operation assigment based using `OperatorAssignmentBasedAttributes`.
   4. From every operation assignment, generate the column.
   5. Generate all the meta-features that are Value based using `OperatorAssignmentBasedAttributes`.

`OperatorAssignmentBasedAttributes` generates meta-features that are "parent dependent" and don't require
us to generate the values of the new attribute. 
- characteristic about the Operator.
- statistical test on source and target columns.
- statistical test of source and target against other columns in dataset.

`DatasetBasedAttributes` creates the dataset meta-features for the dataset.
1. General info.
2. Evaluation info (Auc, PR, logloss,...)
3. Entropy - Information gain (avg, min, max,...)
4. Statistical tests (TTest, ChiSqaureTest)


we write each type of features in seperate file under ''


