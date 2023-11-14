import logging

from torch import nn

from .blocks import Spatial_Basic_Block, Temporal_Basic_Block, Temporal_MultiScale_Block, Spatial_AAGCN_Block, \
    Spatial_CTRGCN_Block


class ResGCN_Module(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_block, temporal_block, A, initial=False, stride=1,
                 kernel_size=[9, 2], **kwargs):
        super(ResGCN_Module, self).__init__()

        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if initial:
            module_res, block_res = False, False
        elif spatial_block == 'Basic' and temporal_block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if spatial_block == "Basic":
            spatial_block = Spatial_Basic_Block
        elif spatial_block == "AAGCN":
            spatial_block = Spatial_AAGCN_Block
        elif spatial_block == "CTRGCN":
            spatial_block = Spatial_CTRGCN_Block
        if temporal_block == "Basic":
            temporal_block = Temporal_Basic_Block
        elif temporal_block == "MultiScale":
            temporal_block = Temporal_MultiScale_Block

        if initial and 'adaptive' in kwargs:
            kwargs['adaptive'] == False
        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, A, block_res, **kwargs)
        self.tcn = temporal_block(out_channels, temporal_window_size, stride, block_res, **kwargs)

    def forward(self, x):
        return self.tcn(self.scn(x), self.residual(x))


class AttGCN_Module(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_block, temporal_block, A, attention, stride=1,
                 kernel_size=[9, 2], **kwargs):
        super(AttGCN_Module, self).__init__()

        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if spatial_block == 'Basic' and temporal_block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if spatial_block == "Basic":
            spatial_block = Spatial_Basic_Block
        elif spatial_block == "AAGCN":
            spatial_block = Spatial_AAGCN_Block
        elif spatial_block == "CTRGCN":
            spatial_block = Spatial_CTRGCN_Block
        if temporal_block == "Basic":
            temporal_block = Temporal_Basic_Block
        elif temporal_block == "MultiScale":
            temporal_block = Temporal_MultiScale_Block

        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, A, block_res, **kwargs)
        self.tcn = temporal_block(out_channels, temporal_window_size, stride, block_res, **kwargs)
        self.att = attention(out_channels, **kwargs)

    def forward(self, x):
        return self.att(self.tcn(self.scn(x), self.residual(x)))
