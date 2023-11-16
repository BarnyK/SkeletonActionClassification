import torch

from models.stgcpp.stgcn import STGCN
from models.tpgcn import TPGCN, __structure, __reduction, __attention, AttGCN_Module, Graph


def create_stgcnpp(num_classes: int, channels: int, skeleton_type: str) -> STGCN:
    graph_cfg = {'layout': skeleton_type, 'mode': 'spatial'}
    model = STGCN(graph_cfg, num_classes, in_channels=channels, gcn_adaptive="init", gcn_with_res=True,
                  tcn_type='mstcn')
    return model


def create_tpgcn(num_clesses: int, branches: int, channels: int, skeleton_type: str, labeling: str = "distance",
                 graph_type: str = "mutual"):
    data_shape = [branches, channels, 0, 0, 0]
    if labeling not in ["distance", "spatial"]:
        raise KeyError(f"{labeling} labeling not supported")
    if graph_type not in ["mutual", "mutual-inter"]:
        raise KeyError(f"{graph_type} graph type not supported")

    g = Graph(skeleton_type, graph_type, labeling)

    model = TPGCN(**(__structure["m19"]), **(__reduction["r1"]), kernel_size=[3, 2], reduct_ratio=2, bias=True,
                  data_shape=data_shape, num_class=num_clesses, A=torch.Tensor(g.A), parts=g.parts,
                  module=AttGCN_Module,
                  attention=__attention["stpa"])
    return model
