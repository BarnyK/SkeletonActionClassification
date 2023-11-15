import logging
import os

import numpy as np

from shared.skeletons import ntu, ntu_coco, coco


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class Graph:
    def __init__(self, layout, graph, labeling, num_person_out=1, max_hop=10, dilation=1, normalize=True,
                 threshold=0.2, **kwargs):
        self.layout = layout
        self.labeling = labeling
        self.graph = graph
        if labeling not in ['spatial', 'distance', 'zeros', 'ones', 'eye', 'pairwise0', 'pairwise1', 'geometric']:
            logging.info('')
            logging.error('Error: Do NOT exist this graph labeling: {}!'.format(self.labeling))
            raise ValueError()
        self.normalize = normalize
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_person_out = num_person_out
        self.threshold = threshold

        # get edges
        self.num_node, self.edge, self.parts, self.center = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.layout == "coco17":
            if self.graph == 'physical':
                num_node = coco.num_nodes
                neighbor_link = coco.edges
                parts = coco.parts
                center = coco.center
            elif self.graph == 'mutual':
                num_node = coco.num_nodes * 2
                neighbor_link = coco.edges + [(x + coco.num_nodes, y + coco.num_nodes) for x, y in coco.edges]
                neighbor_link += [(coco.center, coco.center + ntu_coco.num_nodes) ] # Link between centers
                parts = coco.parts + [x + coco.num_nodes for x in coco.parts]
                center = coco.center
            else:
                raise ValueError()
        elif self.layout == "ntu_coco":
            if self.graph == 'physical':
                num_node = ntu_coco.num_nodes
                neighbor_link = ntu_coco.edges
                parts = ntu_coco.parts
                center = ntu_coco.center
            elif self.graph == 'mutual':
                num_node = ntu_coco.num_nodes * 2
                neighbor_link = ntu_coco.edges + [(x + ntu_coco.num_nodes, y + ntu_coco.num_nodes) for x, y in
                                                  ntu_coco.edges]
                neighbor_link += [(ntu_coco.center, ntu_coco.center + ntu_coco.num_nodes) ] # Link between centers
                parts = ntu_coco.parts + [x + ntu_coco.num_nodes for x in ntu_coco.parts]
                center = ntu_coco.center
            else:
                raise ValueError()
        elif self.layout == "ntu":
            if self.graph == 'physical':
                num_node = ntu.num_nodes
                neighbor_link = ntu.edges
                parts = ntu.parts
                center = ntu.center
            elif self.graph == 'mutual':
                num_node = ntu.num_nodes * 2
                neighbor_link = ntu.edges + [(x + ntu.num_nodes, y + ntu.num_nodes) for x, y in ntu.edges]
                neighbor_link += [(ntu.center, ntu.center + ntu.num_nodes)]
                parts = ntu.parts + [x + ntu.num_nodes for x in ntu.parts]
                center = ntu.center
            elif self.graph == 'mutual-inter':
                num_node = ntu.num_nodes * 2
                neighbor_link = ntu.edges + [(x + ntu.num_nodes, y + ntu.num_nodes) for x, y in ntu.edges]
                neighbor_link += [(ntu.center, ntu.center + ntu.num_nodes)]
                neighbor_link += [(22, 24), (47, 49), (22, 47), (24, 49)]
                parts = ntu.parts + [x + ntu.num_nodes for x in ntu.parts]
                center = 21 - 1
            else:
                raise ValueError()
        else:
            logging.info('')
            logging.error('Error: Do NOT exist this dataset: {}!'.format(self.layout))
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, parts, center

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        self.oA = A
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):

        if self.labeling == 'distance':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        elif self.labeling == 'spatial':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if hop_dis[j, i] == hop:
                            # if hop_dis[j, self.center] == np.inf or hop_dis[i, self.center] == np.inf:
                            #     continue
                            if hop_dis[j, self.center] == hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif hop_dis[j, self.center] > hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)

        elif self.labeling == 'zeros':
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))

        elif self.labeling == 'ones':
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.ones((len(valid_hop), self.num_node, self.num_node))
            for i in range(len(valid_hop)):
                A[i] = self._normalize_digraph(A[i])


        elif self.labeling == 'eye':

            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i in range(len(valid_hop)):
                A[i] = self._normalize_digraph(np.eye(self.num_node, self.num_node))

        elif self.labeling == 'pairwise0':
            # pairwise0: only pairwise inter-body link
            assert 'mutual' in self.graph

            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            v = self.num_node // 2
            for i in range(len(valid_hop)):
                A[i, v:, :v] = np.eye(v, v)
                A[i, :v, v:] = np.eye(v, v)
                A[i] = self._normalize_digraph(A[i])


        elif self.labeling == 'pairwise1':
            assert 'mutual' in self.graph
            v = self.num_node // 2
            self.edge += [(i, i + v) for i in range(v)]
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        elif self.labeling == 'geometric':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1

            geometric_matrix = np.load(os.path.join(os.getcwd(), 'src/dataset/a.npy'))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if geometric_matrix[i, j] > self.threshold:
                        adjacency[i, j] += geometric_matrix[i, j]
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD
