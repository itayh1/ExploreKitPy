from typing import List

import numpy as np
import pandas as pd

from Data.Dataset import Dataset
from Evaluation.DatasetBasedAttributes import DatasetBasedAttributes
from Evaluation.OperatorAssignmentBasedAttributes import OperatorAssignmentBasedAttributes
from Properties import Properties
from RL.Env.Environment import Environment, ActionResult

Action = int
class ExploreKitEnv(Environment):

    def __int__(self, dataset: Dataset, features: List[pd.Series]):
        dba = DatasetBasedAttributes()
        oaba = OperatorAssignmentBasedAttributes()
        dataset_based_features = dba.getDatasetBasedFeatures(dataset, Properties.classifier)
        parent_based_features = oaba.getOperatorAssignmentBasedMetaFeatures(dataset)
        self.features = features
        self.actions_space = len(features) + len(Properties.unaryOperators.split(',')) \
                             + len(Properties.nonUnaryOperators.split(','))

        self.state = np.concatenate([dataset_based_features, np.zeros((28,))])

    def reset(self):
        pass

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
        if action < len(self.features):
            pass
        elif len(self.features) <= action:



    def _action_select_feature(self, action: Action):
        pass

    def _action_select_operation(self, action: Action):
        pass
