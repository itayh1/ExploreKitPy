import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from Env.Environment import EnvGym
from PytorchDQN import DQN


def main():
    env = EnvGym()
    model = DQN(env)
    model.train()

if __name__ == '__main__':
    main()