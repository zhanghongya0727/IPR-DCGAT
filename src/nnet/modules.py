#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, Fenia Christopoulou, National Centre for Text Mining,
# School of Computer Science, The University of Manchester.
# https://github.com/fenchri/edge-oriented-graph/

import torch
from torch import nn, torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nnet.gcn import GraphConvolution_old, GraphAttentionLayer_old


class EmbedLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout, ignore=None, freeze=False):
        super(EmbedLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.freeze = freeze
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=ignore)
        self.embedding.weight.requires_grad = not freeze
        self.drop = nn.Dropout(dropout)

    def forward(self, xs):
        embeds = self.embedding(xs)
        if self.drop.p > 0:
            embeds = self.drop(embeds)
        return embeds


class Encoder_rnn(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_embedding, dropout_encoder):
        super(Encoder_rnn, self).__init__()
        self.wordEmbeddingDim = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size, self.hidden_size, bidirectional=True)
        self.dropout_embedding = nn.Dropout(p=dropout_embedding)
        self.dropout_encoder = nn.Dropout(p=dropout_encoder)

    def forward(self, seq, lens):
        batch_size = seq.shape[0]
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)
        seq_embd = self.dropout_embedding(seq_)
        packed = pack_padded_sequence(seq_embd, lens_sorted, batch_first=True)
        self.encoder.flatten_parameters()
        output, h = self.encoder(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output.contiguous()
        output = torch.index_select(output, 0, lens_argsort_argsort)  # B x m x 2l
        h = h.permute(1, 0, 2).contiguous().view(batch_size, 1, -1)
        h = torch.index_select(h, 0, lens_argsort_argsort)
        output = self.dropout_encoder(output)
        h = self.dropout_encoder(h)
        return output, h


class Encoder(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers, bidirectional, dropout):
        super(Encoder, self).__init__()
        self.enc = nn.LSTM(input_size=input_size,
                           hidden_size=rnn_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.feature_size = rnn_size
        if bidirectional:
            self.feature_size *= 2

    @staticmethod
    def sort(lengths):
        sorted_len, sorted_idx = lengths.sort()  # indices that result in sorted sequence
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(lengths.size(0) - 1, 0, lengths.size(0)).long()  # for big-to-small
        return sorted_idx, original_idx, reverse_idx

    def forward(self, embeds, lengths, hidden=None):
        sorted_idx, original_idx, reverse_idx = self.sort(lengths)
        embeds = nn.utils.rnn.pad_sequence(embeds, batch_first=True, padding_value=0)
        embeds = embeds[sorted_idx][reverse_idx]  # big-to-small
        packed = pack_padded_sequence(embeds, list(lengths[sorted_idx][reverse_idx].data), batch_first=True)
        self.enc.flatten_parameters()
        out_packed, _ = self.enc(packed, hidden)
        outputs, _ = pad_packed_sequence(out_packed, batch_first=True)
        outputs = self.drop(outputs)
        outputs = outputs[reverse_idx][original_idx][reverse_idx]
        return outputs


class Classifier(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(in_features=in_size,
                             out_features=out_size,
                             bias=True)

    def forward(self, xs):
        if self.drop.p > 0:
            xs = self.drop(xs)
        xs = self.lin(xs)
        return xs


class Sentence(nn.Module):
    def __init__(self, input_size):
        super(Sentence, self).__init__()
        self.W = nn.Parameter(nn.init.normal_(torch.empty(input_size, input_size)), requires_grad=True)
        self.u = nn.Parameter(nn.init.normal_(torch.empty(input_size)), requires_grad=True)
        self.bias = nn.Parameter(nn.init.normal_(torch.empty(input_size)), requires_grad=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        att = torch.matmul(input, self.W)
        att = torch.add(att, self.bias)
        att = self.tanh(att)
        att = torch.matmul(att, self.u.t())
        att = self.sigmoid(att).unsqueeze(1)
        sents = torch.matmul(att, input).squeeze(1)
        return sents


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):  # 1_layer or 2 layer
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution_old(nfeat, nhid)
        self.gc2 = GraphConvolution_old(nfeat, nhid)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj_):
        x = self.drop(F.relu(self.gc1(x, adj_)))
        x = self.drop(F.relu(self.gc2(x, adj_)))
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads, alpha, device, dropout):
        super(GAT, self).__init__()
        self.device = device
        self.attentions = [
            GraphAttentionLayer_old(nfeat, nhid, alpha=alpha, concat=True, device=self.device, dropout=dropout)
            for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.drop(x)
        stack = torch.stack([att(x, adj) for att in self.attentions])  # average
        x = torch.mean(stack, 0)
        x = self.drop(x)
        x = F.elu(x)
        return x
