import torch

AVAILABLE_LOSS = ['cross_entropy']


def build_loss(cfg):
    "build losses in right order"
    assert cfg.loss.name in AVAILABLE_LOSS
    if cfg.loss.name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()