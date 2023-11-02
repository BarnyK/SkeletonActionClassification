from torch import nn


class Head(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
