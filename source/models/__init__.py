from models.stgcpp.stgcn import STGCN
import torch

def create_stgcnpp():
    graph_cfg = {'layout': 'coco', 'mode': 'spatial'}
    num_classes = 60
    model = STGCN(graph_cfg, num_classes, in_channels=7, gcn_adaptive="init", gcn_with_res=True, tcn_type='mstcn')
    return model


if __name__ == "__main__":
    import torch
    import time
    model = create_stgcnpp()

    device = torch.device("cuda:0")

    model.to(device)
    xd = torch.rand((5, 2, 64, 17, 7),device=device)

    for i in range(100):
        st = time.time()
        ababa = model(xd)
        et = time.time()
        print(et - st)
        print(ababa.shape)
        break