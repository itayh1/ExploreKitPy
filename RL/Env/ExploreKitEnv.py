from typing import List

import numpy as np
import pandas as pd

from Data.Dataset import Dataset
from Data.Fold import Fold
from Evaluation.DatasetBasedAttributes import DatasetBasedAttributes
from Evaluation.OperatorAssignment import OperatorAssignment
from Evaluation.OperatorAssignmentBasedAttributes import OperatorAssignmentBasedAttributes
from Evaluation.OperatorsAssignmentsManager import OperatorsAssignmentsManager
from Operators.Operator import Operator
from Properties import Properties
from RL.Env.Environment import Environment, ActionResult

Action = int
class ExploreKitEnv(Environment):

    def __int__(self, dataset: Dataset, features: List[pd.Series]):
        dba = DatasetBasedAttributes()
        dataset_based_features = dba.getDatasetBasedFeatures(dataset, Properties.classifier)
        parent_based_features = self._get_parentbase_features(dataset, features)
        inbetween_base_features = self._get_inbetween_base_features(features)

        self.dataset = dataset
        self.features = features
        self.unary_operators = OperatorsAssignmentsManager.getUnaryOperatorsList()
        self.non_unary_operators = OperatorsAssignmentsManager.getNonUnaryOperatorsList()
        self.actions_space = len(features) + len(self.unary_operators) + len(self.non_unary_operators)

        self.state = np.concatenate([dataset_based_features, parent_based_features, inbetween_base_features])

        self.selected_features = []

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
        elif action < num_of_features + len(self.unary_operators) + len(self.non_unary_operators):
            op = self.non_unary_operators[action - len(self.unary_operators) - num_of_features]
        else:
            raise Exception("action is out of range")

    def _action_for_operation(self, op: Operator) -> ActionResult:
        if len(self.selected_features) == 0:
            return ActionResult(-10, True, self.state)

        sourceColumns = self.selected_features[:-1]
        targetColumns = [self.selected_features[-1]]
        if not op.isApplicable(self.dataset, sourceColumns, targetColumns):
            return ActionResult(-1, True, self.state)

        op.processTrainingSet(self.dataset, sourceColumns, targetColumns)
        new_feature = op.generate(self.dataset, sourceColumns, targetColumns)


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
            df = pd.DataFrame(features[:i] + features[i+1:])
            fold = Fold([0, 1], False)
            dataset = Dataset(df, [fold], '', '', Properties.randomSeed, Properties.maxNumberOfDiscreteValuesForInclusionInSet)
            attributes_infos = oaba.getGeneratedAttributeValuesMetaFeatures(dataset, empty_oa, feature)
            attributes_values = [attr.value for attr in attributes_infos.values()]
            parentbase_features.append(attributes_values)
        return np.concatenate(parentbase_features)

    #endregion
    def _action_select_feature(self, action: Action):
        pass

    def _action_select_operation(self, action: Action):
        pass
