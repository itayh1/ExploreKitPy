import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from PytorchDQN import DQN


def main():
    # model = DQN()
    #
    # trainer = Trainer(
    #     accelerator="auto",
    #     devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    #     max_epochs=1500,
    #     val_check_interval=50,
    #     logger=CSVLogger(save_dir="logs/"),
    # )
    # trainer.fit(model)
    model = DQN()
    model.train()

if __name__ == '__main__':
    main()