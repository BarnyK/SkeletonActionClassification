from models.stgcpp.stgcn import STGCN


def create_stgcnpp(num_classes: int, channels: int, graph_cfg: dict = None):
    if graph_cfg is None:
        graph_cfg = {'layout': 'coco', 'mode': 'spatial'}
    model = STGCN(graph_cfg, num_classes, in_channels=channels, gcn_adaptive="init", gcn_with_res=True,
                  tcn_type='mstcn')
    return model


if __name__ == "__main__":
    import torch
    import time

    model = create_stgcnpp()

    device = torch.device("cuda:0")

    model.to(device)
    xd = torch.rand((5, 2, 64, 17, 7), device=device)

    for i in range(100):
        st = time.time()
        ababa = model(xd)
        et = time.time()
        print(et - st)
        print(ababa.shape)
        break
