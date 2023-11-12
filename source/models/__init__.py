from models.stgcpp.stgcn import STGCN


def create_stgcnpp(num_classes: int, channels: int, skeleton_type: str) -> STGCN:
    graph_cfg = {'layout': skeleton_type, 'mode': 'spatial'}
    model = STGCN(graph_cfg, num_classes, in_channels=channels, gcn_adaptive="init", gcn_with_res=True,
                  tcn_type='mstcn')
    return model


if __name__ == "__main__":
    import torch
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
