# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn

torch.set_printoptions(profile="full")


class WalkLayer(nn.Module):
    def __init__(self, input_size, beta=0.9, iters=0, device=-1):
        super(WalkLayer, self).__init__()
        # self.V = nn.Parameter(nn.init.normal_(torch.empty(input_size, input_size)), requires_grad=True)
        self.W = nn.Parameter(nn.init.normal_(torch.empty(input_size, input_size)), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.beta = beta
        self.device = device
        self.iters = iters

    def generate(self, old_graph):
        graph = torch.matmul(old_graph, self.W[None, None])  # (B, I, I, D) 表示W*ekj shape ([3,72,72,100])
        graph = torch.einsum('bijk, bjmk -> bijmk', graph, old_graph)
        return old_graph, graph

    def generate_2(self, old_graph):
        graph_sig = torch.matmul(old_graph, self.W[None, None])  # (B, I, I, D) 表示W*ekj shape ([3,72,72,100])
        graph_sig = torch.einsum('bijk, bjmk -> bijmk', graph_sig, old_graph)
        graph_v = torch.matmul(old_graph, self.V[None, None])  # 多加的另一个V*ekj
        graph_v = torch.einsum('bijk, bjmk -> bijmk', graph_v, old_graph)
        return old_graph, graph_sig, graph_v

    @staticmethod
    def mask_invalid_paths_o(graph, mask3d):
        items = range(graph.size(1))
        graph[:, :, items, items] = float('0.0')
        graph[:, items, items] = float('0.0')
        graph[:, items, :, items] = float('0.0')
        graph = torch.where(mask3d.unsqueeze(-1), graph, torch.as_tensor([float('0.0')]).to(graph.device))
        graph = torch.where(torch.eq(graph, 0.0).all(dim=4, keepdim=True),
                            torch.as_tensor([float('0.0')]).to(graph.device),
                            graph)
        return graph

    @staticmethod
    def mask_invalid_paths(graph, mask3d):
        items = range(graph.size(1))
        graph[:, :, items, items] = float('-inf')
        graph[:, items, items] = float('-inf')
        graph[:, items, :, items] = float('-inf')
        graph = torch.where(mask3d.unsqueeze(-1), graph, torch.as_tensor([float('-inf')]).to(graph.device))
        graph = torch.where(torch.eq(graph, 0.0).all(dim=4, keepdim=True),
                            torch.as_tensor([float('-inf')]).to(graph.device),
                            graph)
        return graph

    def aggregate_relu(self, old_graph, new_graph):
        beta_mat = torch.where(torch.eq(new_graph, 0.0).all(dim=2),
                               torch.ones_like(old_graph),
                               torch.full_like(old_graph, self.beta))
        new_graph = self.relu(new_graph)
        new_graph = torch.sum(new_graph, dim=2)
        new_graph = torch.lerp(new_graph, old_graph, weight=beta_mat)
        return new_graph

    def aggregate_tan(self, old_graph, new_graph):
        beta_mat = torch.where(torch.eq(new_graph, 0.0).all(dim=2),
                               torch.ones_like(old_graph),
                               torch.full_like(old_graph, self.beta))
        new_graph = self.tanh(new_graph)
        new_graph = torch.sum(new_graph, dim=2)
        new_graph = torch.lerp(new_graph, old_graph, weight=beta_mat)
        return new_graph

    def aggregate_sig(self, old_graph, new_graph):
        beta_mat = torch.where(torch.isinf(new_graph).all(dim=2),
                               torch.ones_like(old_graph),
                               torch.full_like(old_graph, self.beta))
        new_graph = self.sigmoid(new_graph)
        new_graph = torch.sum(new_graph, dim=2)
        new_graph = torch.lerp(new_graph, old_graph, weight=beta_mat)
        return new_graph

    def aggregate_gtu(self, old_graph, graph_sin, graph_tan):
        beta_mat = torch.where(torch.isinf(graph_sin).all(dim=2),
                               torch.ones_like(old_graph),
                               torch.full_like(old_graph, self.beta))
        graph_sin = self.sigmoid(graph_sin)
        graph_tan = self.tanh(graph_tan)
        new_graph = torch.mul(graph_sin, graph_tan)
        new_graph = torch.sum(new_graph, dim=2)
        new_graph = torch.lerp(new_graph, old_graph, weight=beta_mat)
        return new_graph

    def aggregate_glu(self, old_graph, graph_sig, graph):
        beta_mat = torch.where(torch.isinf(graph_sig).all(dim=2),
                               torch.ones_like(old_graph),
                               torch.full_like(old_graph, self.beta))
        graph_sig = self.sigmoid(graph_sig)
        new_graph = torch.mul(graph_sig, graph)
        new_graph = torch.sum(new_graph, dim=2)
        new_graph = torch.lerp(new_graph, old_graph, weight=beta_mat)
        return new_graph

    def forward(self, graph, mask_=None, activate=None):
        if activate == 'sigmoid':
            old_graph, graph_sig = self.generate(graph)  # sigmoid W
            graph_sig = self.mask_invalid_paths(graph_sig, mask_)
            graph = self.aggregate_sig(old_graph, graph_sig)
        elif activate == 'glu':
            old_graph, graph_sig, graph = self.generate_2(graph)  # glu W, V
            graph_sig = self.mask_invalid_paths(graph_sig, mask_)
            graph = self.aggregate_glu(old_graph, graph_sig, graph)
        elif activate == 'gtu':
            old_graph, graph_sig, graph_tan = self.generate_2(graph)  # gtu W, V
            graph_sig = self.mask_invalid_paths(graph_sig, mask_)
            graph = self.aggregate_gtu(old_graph, graph_sig, graph_tan)
        elif activate == 'tanh':
            old_graph, graph_tan = self.generate(graph)  # tan W
            graph_tan = self.mask_invalid_paths_o(graph_tan, mask_)
            graph = self.aggregate_tan(old_graph, graph_tan)
        elif activate == 'relu':
            old_graph, graph_relu = self.generate(graph)  # relu W
            graph_relu = self.mask_invalid_paths_o(graph_relu, mask_)
            graph = self.aggregate_relu(old_graph, graph_relu)
        else:
            print('error')
        return graph
