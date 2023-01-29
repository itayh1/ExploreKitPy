from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from Data.Dataset import Dataset
from Data.Fold import Fold
from Evaluation.Classifier import Classifier
from Evaluation.DatasetBasedAttributes import DatasetBasedAttributes
from Evaluation.OperatorAssignment import OperatorAssignment
from Evaluation.OperatorAssignmentBasedAttributes import OperatorAssignmentBasedAttributes
from Evaluation.OperatorsAssignmentsManager import OperatorsAssignmentsManager
from Operators.Operator import Operator
from Properties import Properties
from RL.Env.Environment import Environment, ActionResult

Action = int
class ExploreKitEnv(Environment):

    def __init__(self, dataset: Dataset, features: List[pd.Series], classifier_name: str):
        dba = DatasetBasedAttributes()
        dataset_based_features = [attr_info.value for attr_info in dba.getDatasetBasedFeatures(dataset, Properties.classifier).values()]
        parent_based_features = self._get_parentbase_features(dataset, features)
        inbetween_base_features = self._get_inbetween_base_features(features)

        self.dataset = dataset
        self.features = features
        self.unary_operators = OperatorsAssignmentsManager.getUnaryOperatorsList()
        self.non_unary_operators = OperatorsAssignmentsManager.getNonUnaryOperatorsList()
        self.actions_space = len(features) + len(self.unary_operators) + len(self.non_unary_operators)

        self.state = np.concatenate([dataset_based_features, parent_based_features, inbetween_base_features])

        self.selected_features = []

        # classifier
        self.classifier_name = classifier_name
        self.baseline_score = self._get_score(self.dataset.df)

    def reset(self):
        self.selected_features = []

    def sample(self) -> ActionResult:
        random_action = np.random.randint(self.actions_space)
        return self.action(random_action)

    def get_action_space(self) -> int:
        ''':return number of actions exists'''
        return self.actions_space

    def get_observation_space(self) -> int:
        ''':return size of observation (state) space'''
        return self.state.shape[0]

    def action(self, action: Action) -> ActionResult:
        num_of_features = len(self.features)
        if action < num_of_features:
            self.selected_features.append(self.features[action])
            return ActionResult(0.0, False, self.state)

        elif action < num_of_features + len(self.unary_operators):
            op = self.unary_operators[action - num_of_features]
            return self._action_for_operation(op)

        elif action < num_of_features + len(self.unary_operators) + len(self.non_unary_operators):
            op = self.non_unary_operators[action - len(self.unary_operators) - num_of_features]
            return self._action_for_operation(op)

        raise Exception("action is out of range")

    #region Action
    def _action_for_operation(self, op: Operator) -> ActionResult:
        if len(self.selected_features) == 0:
            return ActionResult(-10, True, self.state)

        source_columns = self.selected_features[:-1]
        target_columns = [self.selected_features[-1]]
        if not op.isApplicable(self.dataset, source_columns, target_columns):
            return ActionResult(-1, True, self.state)

        op.processTrainingSet(self.dataset, source_columns, target_columns)
        new_feature = op.generate(self.dataset, source_columns, target_columns)
        df = self.dataset.df.copy()
        df['new_feature'] = new_feature
        score = self._get_score(df)

        if score > self.baseline_score:
            return ActionResult(10, True, self.state)
        else:
            return ActionResult(-1, True, self.state)

    def _get_score(self, df: pd.DataFrame):
        classifier = Classifier(self.classifier_name)
        classifier.buildClassifier(df.iloc[self.dataset.getIndicesOfTrainingInstances(),:])
        test_set = df.iloc[self.dataset.getIndicesOfTestInstances(), :]
        evaluation_info = classifier.evaluateClassifier(test_set)
        score = roc_auc_score(test_set[self.dataset.targetClass],
                              evaluation_info.getScoreDistribution()[:, 1])
        return score

    #endregion

    #region BuildState
    def _get_parentbase_features(self, dataset: Dataset, features: List[pd.Series]):
        oaba = OperatorAssignmentBasedAttributes()
        empty_oa = OperatorAssignment([], [], Operator(), None)
        parentbase_features = []
        for feature in features:
            attributes_infos = oaba.getGeneratedAttributeValuesMetaFeatures(dataset, empty_oa, feature)
            attributes_values = [attr.value for attr in attributes_infos.values()]
            parentbase_features.append(attributes_values)
        return np.concatenate(parentbase_features)

    def _get_inbetween_base_features(self, features: List[pd.Series]):
        oaba = OperatorAssignmentBasedAttributes()
        empty_oa = OperatorAssignment([], [], Operator(), None)
        parentbase_features = []
        for i, feature in enumerate(features):
            df = pd.DataFrame(features[:i] + features[i+1:] + [features[i]]).T
            fold = Fold(feature.unique().tolist(), False)
            dataset = Dataset(df, [fold], feature.name, '', Properties.randomSeed, Properties.maxNumberOfDiscreteValuesForInclusionInSet)
            attributes_infos = oaba.getGeneratedAttributeValuesMetaFeatures(dataset, empty_oa, feature)
            attributes_values = [attr.value for attr in attributes_infos.values()]
            parentbase_features.append(attributes_values)
        return np.concatenate(parentbase_features)

    #endregion
