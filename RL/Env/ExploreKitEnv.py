from typing import List

import pandas as pd

from Data.Dataset import Dataset
from Evaluation.DatasetBasedAttributes import DatasetBasedAttributes
from Properties import Properties
from RL.Env.Environment import Environment, ActionResult


class ExploreKitEnv(Environment):

    def __int__(self, dataset: Dataset, features: List[pd.Series]):
        dba = DatasetBasedAttributes()
        dba_features = dba.getDatasetBasedFeatures(dataset, Properties.classifier)


    def reset(self):
        pass

    def sample(self):
        pass

    def get_action_space(self):
        pass

    def get_observation_space(self):
        pass

    def action(self, action) -> ActionResult:
        pass
