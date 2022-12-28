from typing import List

import gym
import torch.cuda
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam

from Net import Net
from Agent import Agent
from Structures import RLDataset, Experience


class DQN():
    def __init__(self, env: str = "CartPole-v1", lr: float=3e-4, gamma: float = 1.00, epsilon: float = 1.0, epsilon_decay_rate: float = 0.9999, sync_rate: int = 25):
        self.env = gym.make(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.lr = lr
        self.sync_rate = sync_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.net = Net(obs_size, n_actions, 42).to(self.device)
        self.target_net = Net(obs_size, n_actions,42).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = Adam(self.net.parameters(), lr=self.lr)

        self.dataset = RLDataset()
        self.agent = Agent(self.env, self.dataset)

        self.rewards: List[int] = []

    def train(self):
        global_step = 0

        for episode in range(1000):
            episode_reward = 0
            steps = 0
            episode_loss = 0.0
            # step through environment with agent
            while True:
                epsilon = self.get_epsilon()
                exp = self.agent.play_step(self.net, epsilon, self.device)

                loss = self.dqn_mse_loss(exp)
                self.optimizer.zero_grad()
                loss.backward()
                episode_reward += exp.reward
                episode_loss += loss.float()

                if exp.done:
                    # print(self.episode_reward)
                    self.rewards.append(episode_reward)
                    break

                if global_step % self.sync_rate == 0:
                    self.target_net.load_state_dict(self.net.state_dict())

                global_step += 1
                steps += 1

            self.update_stats(episode_loss/steps, episode_reward, episode)

    def dqn_mse_loss(self, exp: Experience) -> nn.MSELoss:
        state, action, reward, done, next_state = exp

        state_action_value = self.net(state)[action] #.gather(1, action).squeeze(-1)

        with torch.no_grad():
            next_state_value = self.target_net(next_state).max(0)[0]
            next_state_value[done] = 0
            next_state_value = next_state_value.detach()

        expected_state_action_value = next_state_value * self.gamma + reward
        return nn.MSELoss()(state_action_value, expected_state_action_value)

    def get_epsilon(self) -> float:
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay_rate)
        return self.epsilon

    def update_stats(self, loss_value, reward, episode):
        print(f'{loss_value=}, {reward=}, {episode=}', end='\r')

    def plot(self):
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.rewards)

