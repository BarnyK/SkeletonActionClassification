from stgcpp.stgcn import STGCN


def create_stgcnpp():
    args = {
        'type': 'STGCN',
        'gcn_adaptive': 'init',
        'gcn_with_res': True,
        'tcn_type': 'mstcn',
        'graph_cfg': {
            'layout': 'coco',
            'mode': 'spatial'
        },
        'pretrained': None
    }
    graph_cfg = {'layout': 'coco', 'mode': 'spatial'}
    model = STGCN(graph_cfg, gcn_adaptive="init", gcn_with_res=True, tcn_type='mstcn')


if __name__ == "__main__":
    create_stgcnpp()
