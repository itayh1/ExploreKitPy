from collections import namedtuple
from typing import NamedTuple
import gym
import numpy as np

ActionResult = NamedTuple(
    "ActionResult",
    fields=[("reward", float), ("done", bool), ("new_state", np.ndarray)],
)

class Environment:
    def get_observation_space(self):
        raise Exception('Not implemented')

    def get_action_space(self):
        raise Exception('Not implemented')

    def action(self, action) -> ActionResult:
        raise Exception('Not implemented')

    def sample(self):
        raise Exception('Not implemented')

    def reset(self):
        raise Exception('Not implemented')


class EnvGym(Environment):
    def __init__(self, env: str = "CartPole-v1"):
        self.env = gym.make(env)

    def get_observation_space(self):
        return self.env.observation_space.shape[0]

    def get_action_space(self):
        return self.env.action_space.n

    def action(self, action) -> ActionResult:
        new_state, reward, done, _, _ = self.env.step(action)
        return ActionResult(reward=reward, done=done, new_state=new_state)

    def sample(self):
        return self.env.action_space.sample()

    def reset(self):
        return self.env.reset()[0]
