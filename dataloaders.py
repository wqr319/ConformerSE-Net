from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from datasets import *

def build_vbd_dataloaders(cfg):
    train_loader = DataLoader(VBDDataset(cfg, 'train'),
                              batch_size=cfg['batch_size'],
                              shuffle=True,
                              num_workers=cfg['WORKERS'],
                              drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(VBDDataset(cfg, 'val'),
                              batch_size=cfg['batch_size'],
                              shuffle=False,
                              num_workers=cfg['WORKERS'],
                              drop_last=True,
                              pin_memory=True)
    return train_loader, val_loader, len(train_loader)

def build_on_fly_dataloaders(cfg):
    train_loader = DataLoader(OnFlyDataset(cfg, 'train'),
                              batch_size=cfg['batch_size'],
                              shuffle=True,
                              num_workers=cfg['WORKERS'],
                              drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(OnFlyDataset(cfg, 'val'),
                              batch_size=cfg['batch_size'],
                              shuffle=False,
                              num_workers=cfg['WORKERS'],
                              drop_last=True,
                              pin_memory=True)
    return train_loader, val_loader, len(train_loader)