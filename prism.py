import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

FLAT_SIZE = 1536

class Prism(nn.Module):
    def __init__(self):
        super(Prism, self).__init__()

        def gen_layers():
            yield nn.Conv2d(3, 32, 3, stride=2)
            yield nn.LeakyReLU()
            yield nn.MaxPool2d(3, 3)
            yield nn.BatchNorm2d(32)
            for l in range(1):
                yield nn.Conv2d(32, 32, 3, stride=2)
                yield nn.LeakyReLU()
                yield nn.MaxPool2d(2, 2)
                yield nn.BatchNorm2d(32)

        self.seq = nn.Sequential(*gen_layers())

    def forward(self, x):
        feats = self.seq(x)
        return feats.reshape(feats.size(0), -1)

class Head(nn.Module):
    def __init__(self, nb_discrete_actions):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(FLAT_SIZE, int(FLAT_SIZE/2)),
            nn.ReLU(),
            nn.Linear(int(FLAT_SIZE/2), nb_discrete_actions),
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
        obs, act, head = batch

        predicted_act = self(obs)
        loss = F.mse_loss(predicted_act, act)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer