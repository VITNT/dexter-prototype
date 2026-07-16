"""OpenHands ST-GCN architecture for the released WLASL2000 checkpoint.

Adapted from AI4Bharat/OpenHands (Apache-2.0), specifically the ST-GCN
encoder, graph partitioning utilities, fully-connected decoder, and network
wrapper. The module is self-contained so serving the checkpoint does not
require the unmaintained OpenHands/PyTorch-Lightning dependency stack.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


WLASL2000_INWARD_EDGES = (
    (2, 0),
    (1, 0),
    (0, 3),
    (0, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (6, 17),
    (7, 8),
    (7, 9),
    (9, 10),
    (7, 11),
    (11, 12),
    (7, 13),
    (13, 14),
    (7, 15),
    (15, 16),
    (17, 18),
    (17, 19),
    (19, 20),
    (17, 21),
    (21, 22),
    (17, 23),
    (23, 24),
    (17, 25),
    (25, 26),
)


def _hop_distance(num_nodes: int, edges: list[tuple[int, int]], max_hop: int) -> np.ndarray:
    adjacency = np.zeros((num_nodes, num_nodes))
    for source, target in edges:
        adjacency[target, source] = 1
        adjacency[source, target] = 1
    hop_distance = np.full((num_nodes, num_nodes), np.inf)
    transfers = [np.linalg.matrix_power(adjacency, distance) for distance in range(max_hop + 1)]
    arrivals = np.stack(transfers) > 0
    for distance in range(max_hop, -1, -1):
        hop_distance[arrivals[distance]] = distance
    return hop_distance


def _normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    degrees = np.sum(adjacency, axis=0)
    inverse_degrees = np.zeros_like(adjacency)
    for index, degree in enumerate(degrees):
        if degree > 0:
            inverse_degrees[index, index] = degree ** -1
    return adjacency @ inverse_degrees


def spatial_adjacency(
    num_nodes: int = 27,
    center: int = 0,
    inward_edges: tuple[tuple[int, int], ...] = WLASL2000_INWARD_EDGES,
    max_hop: int = 1,
    dilation: int = 1,
) -> np.ndarray:
    """Build the three-part spatial graph used by the OpenHands checkpoint."""
    self_edges = [(index, index) for index in range(num_nodes)]
    hop_distance = _hop_distance(num_nodes, self_edges + list(inward_edges), max_hop)
    valid_hops = range(0, max_hop + 1, dilation)
    adjacency = np.zeros((num_nodes, num_nodes))
    for hop in valid_hops:
        adjacency[hop_distance == hop] = 1
    normalized = _normalize_digraph(adjacency)

    partitions: list[np.ndarray] = []
    for hop in valid_hops:
        root = np.zeros((num_nodes, num_nodes))
        close = np.zeros((num_nodes, num_nodes))
        further = np.zeros((num_nodes, num_nodes))
        for row in range(num_nodes):
            for column in range(num_nodes):
                if hop_distance[column, row] != hop:
                    continue
                if hop_distance[column, center] == hop_distance[row, center]:
                    root[column, row] = normalized[column, row]
                elif hop_distance[column, center] > hop_distance[row, center]:
                    close[column, row] = normalized[column, row]
                else:
                    further[column, row] = normalized[column, row]
        if hop == 0:
            partitions.append(root)
        else:
            partitions.extend((root + close, further))
    return np.stack(partitions)


class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(1, 1))

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if adjacency.size(0) != self.kernel_size:
            raise ValueError("Adjacency partition count does not match the graph kernel.")
        convolved = self.conv(inputs)
        batch, channels, frames, vertices = convolved.size()
        convolved = convolved.view(
            batch,
            self.kernel_size,
            channels // self.kernel_size,
            frames,
            vertices,
        )
        output = torch.einsum("nkctv,kvw->nctw", convolved, adjacency)
        return output.contiguous(), adjacency


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        dropout: float = 0,
        residual: bool = True,
    ):
        super().__init__()
        temporal_kernel, spatial_kernel = kernel_size
        padding = ((temporal_kernel - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, spatial_kernel)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (temporal_kernel, 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda inputs: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda inputs: inputs
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.residual(inputs)
        output, adjacency = self.gcn(inputs, adjacency)
        return self.relu(self.tcn(output) + residual), adjacency


class STGCN(nn.Module):
    def __init__(self, in_channels: int = 2, num_nodes: int = 27, n_out_features: int = 256):
        super().__init__()
        adjacency = torch.tensor(spatial_adjacency(num_nodes=num_nodes), dtype=torch.float32)
        self.register_buffer("A", adjacency)
        self.n_out_features = n_out_features
        kernel_size = (9, adjacency.size(0))
        self.data_bn = nn.BatchNorm1d(in_channels * adjacency.size(1))
        self.st_gcn_networks = nn.ModuleList(
            (
                STGCNBlock(in_channels, 64, kernel_size, 1, residual=False),
                STGCNBlock(64, 64, kernel_size, 1),
                STGCNBlock(64, 64, kernel_size, 1),
                STGCNBlock(64, 64, kernel_size, 1),
                STGCNBlock(64, 128, kernel_size, 2),
                STGCNBlock(128, 128, kernel_size, 1),
                STGCNBlock(128, 128, kernel_size, 1),
                STGCNBlock(128, 256, kernel_size, 2),
                STGCNBlock(256, 256, kernel_size, 1),
                STGCNBlock(256, n_out_features, kernel_size, 1),
            )
        )
        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, vertices = inputs.size()
        output = inputs.permute(0, 3, 1, 2).contiguous()
        output = output.view(batch, vertices * channels, frames)
        output = self.data_bn(output)
        output = output.view(batch, vertices, channels, frames)
        output = output.permute(0, 2, 3, 1).contiguous()
        for graph_layer, importance in zip(self.st_gcn_networks, self.edge_importance):
            output, _ = graph_layer(output, self.A * importance)
        output = F.avg_pool2d(output, output.size()[2:])
        return output.view(batch, -1)


class FullyConnectedDecoder(nn.Module):
    def __init__(self, n_features: int = 256, num_classes: int = 2000, dropout_ratio: float = 0.05):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.classifier = nn.Linear(n_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(inputs))


class OpenHandsSTGCN(nn.Module):
    """Exact network topology for the supplied OpenHands WLASL2000 weight."""

    def __init__(self, num_classes: int = 2000):
        super().__init__()
        self.encoder = STGCN(in_channels=2, num_nodes=27, n_out_features=256)
        self.decoder = FullyConnectedDecoder(
            n_features=256,
            num_classes=num_classes,
            dropout_ratio=0.05,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))
