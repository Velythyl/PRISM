import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

FLAT_SIZE = 1536

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
        )

    def forward(self, x):
        #print(self.seq(x).shape)
        return self.seq(x)

class Head(nn.Module):
    def __init__(self, input_space=3136, nb_discrete_actions=None):
        super().__init__()
        assert nb_discrete_actions is not None
        self.seq = nn.Sequential(
            nn.Linear(input_space, int(input_space/2)),
            nn.ReLU(),
            nn.Linear(int(input_space/2), nb_discrete_actions),
            nn.Sigmoid()    # maps to probs, kinda
        )

    def forward(self, x):
        return self.seq(x)

class PrismAndHead(pl.LightningModule):
    def __init__(self, prism, head):
        super(PrismAndHead, self).__init__()
        self.seq = nn.Sequential(prism, head)

    def forward(self, x):
        # prism computes features, and then those are piped to the head to predict actions
        return self.seq(x)

    def predict(self, obs):
        return self(obs)

    def training_step(self, batch, batch_idx):
        obs, act = batch

        predicted_act = self(obs)
        loss = F.cross_entropy(predicted_act, act)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer