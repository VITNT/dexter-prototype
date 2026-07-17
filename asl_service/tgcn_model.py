"""TGCN architecture used by the WLASL pose baseline checkpoint.

The parameter/module names intentionally match the released checkpoint. The
implementation is adapted from ``dxli94/WLASL/code/TGCN/tgcn_model.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GraphConvolutionAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.att = nn.Parameter(torch.empty(55, 55))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.att, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(self.att, support)
        return output if self.bias is None else output + self.bias


class GCBlock(nn.Module):
    def __init__(self, in_features: int, p_dropout: float, is_resi: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_resi = is_resi
        self.gc1 = GraphConvolutionAttention(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(55 * in_features)
        self.gc2 = GraphConvolutionAttention(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(55 * in_features)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.gc1(inputs)
        batch, nodes, features = output.shape
        output = self.bn1(output.view(batch, -1)).view(batch, nodes, features)
        output = self.do(self.act_f(output))
        output = self.gc2(output)
        output = self.bn2(output.view(batch, -1)).view(batch, nodes, features)
        output = self.do(self.act_f(output))
        return output + inputs if self.is_resi else output


class TGCN(nn.Module):
    """Multi-attention temporal graph convolution network for WLASL2000."""

    def __init__(
        self,
        input_feature: int,
        hidden_feature: int,
        num_class: int,
        p_dropout: float,
        num_stage: int = 1,
        is_resi: bool = True,
    ):
        super().__init__()
        self.num_stage = num_stage
        self.gc1 = GraphConvolutionAttention(input_feature, hidden_feature)
        self.bn1 = nn.BatchNorm1d(55 * hidden_feature)
        self.gcbs = nn.ModuleList(
            GCBlock(hidden_feature, p_dropout=p_dropout, is_resi=is_resi)
            for _ in range(num_stage)
        )
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.gc1(inputs)
        batch, nodes, features = output.shape
        output = self.bn1(output.view(batch, -1)).view(batch, nodes, features)
        output = self.do(self.act_f(output))
        for block in self.gcbs:
            output = block(output)
        return self.fc_out(torch.mean(output, dim=1))
