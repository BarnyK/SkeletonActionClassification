import torch

from models.stgcpp.stgcn import STGCN
from models.tpgcn import TPGCN, __structure, __reduction, __attention, AttGCN_Module, Graph


def create_stgcnpp(num_classes: int, channels: int, skeleton_type: str) -> STGCN:
    graph_cfg = {'layout': skeleton_type, 'mode': 'spatial'}
    model = STGCN(graph_cfg, num_classes, in_channels=channels, gcn_adaptive="init", gcn_with_res=True,
                  tcn_type='mstcn')
    return model


def check_graphs():
    layouts = ["coco17", "ntu", "ntu_coco"]
    labelings = ["distance", "spatial", "pairwise0", "pairwise0"]
    for lay in layouts:
        for label in labelings:
            print(lay, label)
            g = Graph(lay, "mutual", label)
            print(g.A.shape)


def create_tpgcn(num_clesses: int, branches: int, channels: int, symmetry: bool, skeleton_type: str):
    data_shape = [branches, channels, 0, 0, 0]
    # Create A and parts
    # layout is skeleton_type
    # 3 skeleton types
    g = Graph(skeleton_type, "mutual", "distance")

    model = TPGCN(**(__structure["m19"]), **(__reduction["r1"]), kernel_size=[3, 2], reduct_ratio=2, bias=True,
                  data_shape=data_shape, num_class=num_clesses, A=torch.Tensor(g.A), parts=g.parts,
                  module=AttGCN_Module,
                  attention=__attention["stpa"])
    return model


if __name__ == "__main__":
    import time

    check_graphs()
    # model = create_tpgcn(11, 4, 6, False, "ntu")

    # device = torch.device("cpu")
    #
    # model.to(device)
    # xd = torch.rand((5, 2, 64, 17, 3), device=device)
    #
    # for i in range(100):
    #     st = time.time()
    #     ababa = model(xd)
    #     et = time.time()
    #     print(et - st)
    #     print(ababa.shape)
    #     break
if __name__ == "__main__2":

    import time

    model = create_stgcnpp(120, 3)

    device = torch.device("cpu")

    model.to(device)
    xd = torch.rand((5, 2, 64, 17, 3), device=device)

    for i in range(100):
        st = time.time()
        ababa = model(xd)
        et = time.time()
        print(et - st)
        print(ababa.shape)
        break
