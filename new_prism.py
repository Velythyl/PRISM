import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class Prism(nn.Module):
    """
    Based on the Nature CNN https://github.com/DLR-RM/stable-baselines3/blob/b8c72a53489c6d80196a1dc168835a2f375b868d/stable_baselines3/common/torch_layers.py#L50
    From

    Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    """
    def __init__(self):
        super(Prism, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU()
        )

    def eval(self):
        super().eval()
        for param in self.parameters():
            param.requires_grad = False

    def train_pred(self, x):
        return self.seq(x)

    def forward(self, x):
        return self.seq(x)

class PrismAndHead(pl.LightningModule):
    def __init__(self, prism, nb_discrete_actions):
        super(PrismAndHead, self).__init__()
        self.prism = prism

        self.seq = nn.Sequential(
            nn.Linear(512, nb_discrete_actions),
            nn.Tanh()
        )

    def forward(self, x):
        # prism computes features, and then those are piped to the head to predict actions
        return self.seq(self.prism(x))

    def predict(self, obs):
        return self(obs)

    def training_step(self, batch, batch_idx):
        obs, act = batch

        predicted_act = self(obs)

        loss = F.cross_entropy(predicted_act, act)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
