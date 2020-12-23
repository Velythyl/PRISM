import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

class PrismBottleneck(nn.Module):
    """
    Based on the Nature CNN https://github.com/DLR-RM/stable-baselines3/blob/b8c72a53489c6d80196a1dc168835a2f375b868d/stable_baselines3/common/torch_layers.py#L50
    From
    Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    """
    def __init__(self, neck):
        super(PrismBottleneck, self).__init__()
        self.neck = neck
        # relu dies https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks

        self.seq1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(3136, 1000),
            nn.LeakyReLU(),
            nn.Dropout(),
        )

        self.seq2 = nn.Sequential(
            nn.Linear(1000, neck),
            nn.ReLU6(),
        )

    def eval(self):
        super().eval()
        for param in self.parameters():
            param.requires_grad = False

    def train_pred(self, x):
        x2 = self.seq1(x)
        return x2, self.seq2(x2)

    def forward(self, x):
        return self.seq2(self.seq1(x))

class PrismBottleneckAndHead(pl.LightningModule):
    def __init__(self, prism, nb_discrete_actions):
        super(PrismBottleneckAndHead, self).__init__()
        self.prism = prism

        self.seq1 = nn.Sequential(
            nn.Linear(32, 1000),
            nn.LeakyReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Linear(1000, nb_discrete_actions),
            nn.LeakyReLU()
        )

    def forward(self, x):
        # prism computes features, and then those are piped to the head to predict actions
        return self.seq2(self.seq1(self.prism(x)))

    def predict(self, obs):
        return self(obs)

    def training_step(self, batch, batch_idx):
        obs, act = batch

        pre_latent, latent = self.prism.train_pred(obs)
        post_latent = self.seq1(latent)
        predicted_act = self.seq2(post_latent)

        il_loss = F.cross_entropy(predicted_act, act)
        reconstruction_loss = F.mse_loss(pre_latent, post_latent)*10

        loss = il_loss + reconstruction_loss
        self.log("il_loss", il_loss)
        self.log("recon_loss", reconstruction_loss)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer