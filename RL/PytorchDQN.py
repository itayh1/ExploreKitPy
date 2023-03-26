from typing import List

import numpy as np
import torch.cuda
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence

from Net import Net
from Agent import Agent
from RL.Env.Environment import Environment
from Structures import RLDataset, Experience, Memory


class DQN():
    def __init__(self, env: Environment, lr: float=3e-4, gamma: float = 1.00, epsilon: float = 1.0,
                 epsilon_decay_rate: float = 0.9999, sync_rate: int = 25, mem_size: int = 300, batch_size: int = 32):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.lr = lr
        self.sync_rate = sync_rate
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs_size = self.env.get_observation_space()
        n_actions = self.env.get_action_space()
        self.net = Net(obs_size, n_actions, 42).to(self.device)
        self.target_net = Net(obs_size, n_actions, 42).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

        # self.dataset = RLDataset()
        self.memory = Memory(self.mem_size)
        self.agent = Agent(self.env, self.memory)

        self.rewards: List[int] = []

    def train(self):
        # Fill memory
        for i in range(0, self.mem_size):
            prev_state = self.env.reset()
            step_count = 0
            local_memory = []
            done = False
            # while step_count < MAX_STEPS:
            while done != True:
                step_count += 1
                action = np.random.randint(0, self.env.get_action_space())
                reward, done, next_state = self.env.action(action)

                local_memory.append((prev_state, action, reward, next_state))

                prev_state = next_state

            self.memory.add_episode(local_memory)

        global_step = 0
        loss_stats = []
        reward_stats = []

        # Start Algorithm
        for episode in range(1000):
            episode_reward = 0
            steps = 0
            episode_loss = 0.0
            local_memory = []
            prev_state = self.env.reset()
            hidden_state, cell_state = self.net.init_hidden_states(bsize=1)
            # step through environment with agent
            while True:
                epsilon = self.get_epsilon()

                if np.random.rand(1) < epsilon:
                    torch_x = torch.from_numpy(prev_state).float().to(self.device)
                    torch_x = torch_x.view(1, 1, -1)
                    torch_x, _ = self._pad_seqequence(torch_x)
                    model_out = self.net.forward(torch_x, bsize=1,  hidden_state=hidden_state,
                                                   cell_state=cell_state)
                    action = np.random.randint(0, self.env.get_action_space())
                    hidden_state = model_out[1][0]
                    cell_state = model_out[1][1]
                else:
                    torch_x = torch.from_numpy(prev_state).float().to(self.device)
                    torch_x = torch_x.view(1, 1, -1)
                    torch_x, _ = self._pad_seqequence(torch_x)
                    model_out = self.net.forward(torch_x, bsize=1, hidden_state=hidden_state,
                                                   cell_state=cell_state)
                    out = model_out[0]
                    action = int(torch.argmax(out[0]))
                    hidden_state = model_out[1][0]
                    cell_state = model_out[1][1]

                reward, done, next_state = self.env.action(action)
                episode_reward += reward

                local_memory.append((prev_state, action, reward, next_state))

                if (global_step % self.sync_rate) == 0:
                    self.target_net.load_state_dict(self.net.state_dict())

                hidden_batch, cell_batch = self.net.init_hidden_states(bsize=self.batch_size)
                batch = self.memory.get_batch(bsize=self.batch_size)

                current_states = []
                acts = []
                rewards = []
                next_states = []

                for b in batch:
                    cs, ac, rw, ns = [], [], [], []
                    for element in b:
                        cs.append(element[0])
                        # ac.append(element[1])
                        # rw.append(element[2])
                        ns.append(element[3])
                    current_states.append(torch.Tensor(cs))
                    # acts.append(torch.Tensor(ac))
                    # rewards.append(torch.Tensor(rw))
                    acts.append(b[-1][1])
                    rewards.append(b[-1][2])
                    next_states.append(torch.Tensor(ns))

                torch_acts = torch.LongTensor(acts).to(self.device)
                torch_rewards = torch.FloatTensor(rewards).to(self.device)

                torch_current_states, _ = self._pad_seqequence(current_states)
                torch_current_states = torch_current_states.to(self.device)

                torch_next_states, _ = self._pad_seqequence(next_states)
                torch_next_states = torch_next_states.to(self.device)

                Q_next, _ = self.net.forward(torch_next_states, bsize=self.batch_size,
                                                 hidden_state=hidden_batch, cell_state=cell_batch)
                Q_next_max, __ = Q_next.detach().max(dim=1)
                target_values = torch_rewards + (self.gamma * Q_next_max)

                Q_s, _ = self.net.forward(torch_current_states, bsize=self.batch_size,
                                            hidden_state=hidden_batch, cell_state=cell_batch)
                Q_s_a = Q_s.gather(dim=1, index=torch_acts.unsqueeze(dim=1)).squeeze(dim=1)

                loss = self.criterion(Q_s_a, target_values)
                #  save performance measure
                loss_stats.append(loss.item())

                # make previous grad zero
                self.optimizer.zero_grad()

                # backward
                loss.backward()

                # update params
                self.optimizer.step()

                if done:
                    self.rewards.append(episode_reward)
                    break

                global_step += 1
                steps += 1

            self.update_stats(loss.item(), episode_reward, episode)
            reward_stats.append(episode_reward)
            self.memory.add_episode(local_memory)

    def _pad_seqequence(self, batch) -> PackedSequence:
        x_lens = [len(x) for x in batch]
        xx_pad = pad_sequence(batch, batch_first=True, padding_value=0)
        x_packed = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False)
        return x_packed, x_lens

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

