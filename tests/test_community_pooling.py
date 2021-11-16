
import unittest
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from deeprank_gnn.community_pooling import community_detection, community_pooling, community_detection_per_batch


class TestCommunity(unittest.TestCase):

    def setUp(self):
        self.edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5],
                                        [1, 0, 2, 1, 4, 3, 5, 4]], dtype=torch.long)

        self.x = torch.tensor(
            [[0], [1], [2], [3], [4], [5]], dtype=torch.float)
        self.data = Data(x=self.x, edge_index=self.edge_index)
        self.data.pos = torch.tensor(
            np.random.rand(self.data.num_nodes, 3))

    def test_detection_mcl(self):
        c = community_detection(
            self.data.edge_index, self.data.num_nodes, method='mcl')

    def test_detection_louvain(self):
        c = community_detection(
            self.data.edge_index, self.data.num_nodes, method='louvain')

    @unittest.expectedFailure
    def test_detection_error(self):
        c = community_detection(
            self.data.edge_index, self.data.num_nodes, method='xxx')

    def test_detection_per_batch_mcl(self):
        batch = Batch().from_data_list([self.data, self.data])
        c = community_detection_per_batch(
            self.data.edge_index, torch.as_tensor([0, 1, 2, 3, 4, 5]),
            self.data.num_nodes, method='mcl')

    def test_detection_per_batch_louvain(self):
        batch = Batch().from_data_list([self.data, self.data])
        c = community_detection_per_batch(
            self.data.edge_index, torch.as_tensor([0, 1, 2, 3, 4, 5]),
            self.data.num_nodes, method='louvain')

    @unittest.expectedFailure
    def test_detection_per_batch_louvain(self):
        batch = Batch().from_data_list([self.data, self.data])
        c = community_detection_per_batch(
            self.data.edge_index, torch.as_tensor([0, 1, 2, 3, 4, 5]),
            self.data.num_nodes, method='xxxx')

    def test_pooling(self):

        batch = Batch().from_data_list([self.data, self.data])
        cluster = community_detection(
            batch.edge_index, batch.num_nodes)

        _ = community_pooling(cluster, batch)


if __name__ == "__main__":
    unittest.main()
