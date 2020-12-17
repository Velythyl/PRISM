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
        # relu dies https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks

        self.seq1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(3136, 1500),
            nn.LeakyReLU(),
        )

        self.seq2 = nn.Sequential(
            nn.Linear(1500, 300),
            nn.Sigmoid()
        )
        """
        old:
        nn.Linear(3136, 1500),
            nn.LeakyReLU(),
            nn.Linear(1500, 512),
            nn.LeakyReLU(),"""
    def train_pred(self, x):
        x2 = self.seq1(x)
        return x2, self.seq2(x2)

    def forward(self, x):
        #print(x.shape)
        #print(self.seq(x).shape)
        return self.seq2(self.seq1(x))

class Head(nn.Module):
    def __init__(self, input_space=128, nb_discrete_actions=None):
        super().__init__()
        assert nb_discrete_actions is not None


    def forward(self, x):
        #print(x.shape)
        return self.seq(x)

class PrismAndHead(pl.LightningModule):
    def __init__(self, prism, nb_discrete_actions):
        super(PrismAndHead, self).__init__()
        self.prism = prism

        self.seq1 = nn.Sequential(
            nn.Linear(300, 1500),
            nn.LeakyReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Linear(1500, nb_discrete_actions),
            # nn.LeakyReLU(),
            # nn.Linear(3000, nb_discrete_actions),
            nn.Sigmoid()  # maps to probs, kinda
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
        reconstruction_loss = F.mse_loss(pre_latent, post_latent)

        #print(il_loss)
        #print(reconstruction_loss)
        #print(loss)

        loss = il_loss + reconstruction_loss
        self.log("il_loss", il_loss)
        self.log("recon_loss", reconstruction_loss)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer