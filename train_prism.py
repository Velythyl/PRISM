import copy
from time import sleep

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from tqdm import trange

from new_prism import PrismAndHead, Prism
from prism_bottleneck import PrismBottleneck, PrismBottleneckAndHead

BATCH_SIZE = 2048
N_EPOCHS = 1

def make_dataloaders_and_pahs(prism, numpy_path, how_many):
    dataset = np.load(numpy_path)
    obs = dataset["a"]
    act = dataset["b"]

    obs_chunks = np.array_split(obs, how_many)
    act_chunks = np.array_split(act, how_many)

    class PrismDataset(Dataset):
        def __init__(self, obs, act):
            self.obs = obs
            self.act = act
            self.to_tensor = ToTensor()

        def __getitem__(self, idx):
            obs = self.obs[idx]
            act = self.act[idx]

            #print(obs.shape)

            return self.to_tensor(obs).float(), torch.as_tensor(act).long()

        def __len__(self):
            return len(self.obs)

    t = []
    for i in range(how_many):
        t.append(DataLoader(PrismDataset(obs_chunks[i], act_chunks[i]), BATCH_SIZE, num_workers=1))
    return t

def accuracy(loader, pah):
    pah.to("cuda")
    t = []
    for obss, acts in loader:
        with torch.no_grad():
            preds = torch.argmax(pah(obss.to("cuda")), dim=1).cpu()
        t.append((preds == acts).float().mean().cpu().numpy())
    return np.array(t).mean()

def train_prism(env_name):
    prism = PrismBottleneck(32)
    loader_pahs = make_dataloaders_and_pahs(prism, f"./{env_name}/dataset/dataset.npz", 1)

    try:
        for i in trange(N_EPOCHS):
            print(i)
            for loader in loader_pahs:
                pah = PrismBottleneckAndHead(prism, 6)
                print(accuracy(loader, pah))
                trainer = pl.Trainer(gpus=1, max_epochs=2000)
                trainer.fit(pah, loader)
                print(accuracy(loader, pah))
    except KeyboardInterrupt:
        pass
    sleep(10)
    torch.save(prism.state_dict(), f"./{env_name}/prism32.pt")
    return prism
train_prism("PongNoFrameskip-v4")