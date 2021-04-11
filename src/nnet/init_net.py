#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from nnet.modules import Encoder_rnn, Classifier, Sentence, GCN, GAT, EmbedLayer, Encoder
from nnet.walks import WalkLayer
from nnet.gcn import GraphConvLayer, GraphAttentionLayer, MultiGraphAttentionLayer, MultiGraphConvLayer
import numpy as np
import os


class BaseNet(nn.Module):
    def __init__(self, params, sizes=None, maps=None, lab2ign=None):
        super(BaseNet, self).__init__()
        self.device = torch.device("cuda:{}".format(params['gpu']) if params['gpu'] != -1 else "cpu")
        self.edg = ['MM', 'SS', 'ME', 'MS', 'ES', 'EE']
        self.dims = {}   # 边的维度
        # 适合GDA这种随机初始化的？？
        # self.word_embed = EmbedLayer(num_embeddings=sizes['word_size'],
        #                              embedding_dim=params['word_dim'],
        #                              dropout=params['drop_i'],
        #                              ignore=None,
        #                              freeze=params['freeze_words'])
        # self.encoder = Encoder(input_size=params['word_dim'], # can add conference and entity_type
        #                        rnn_size=params['out_dim'],
        #                        num_layers=params['bilstm_layers'],
        #                        bidirectional=True,
        #                        dropout=0.0)
        # 适合CDR这种已经有词向量的的
        data_word_vec = np.load(os.path.join("../embeds/arr.npy"))
        self.word_embed = nn.Embedding(data_word_vec.shape[0], data_word_vec.shape[1])
        self.word_embed.weight.data.copy_(torch.from_numpy(data_word_vec))
        self.word_embed.weight.requires_grad = False
        temp = data_word_vec.shape[1]
        if params['conference']:
            temp += params['conference_dim']
        if params['entity_type']:      
            temp += params['entity_type_dim']
        self.rnn_sent = Encoder_rnn(temp, temp, dropout_embedding=0.2, dropout_encoder=0.3)
        self.linear_re = nn.Linear(temp * 2, params['out_dim'])
        self.dropout_rate = nn.Dropout(params['drop_rate'])
        if params['conference']:
            self.conference_embed = nn.Embedding(512, params['conference_dim'])
        if params['entity_type']:
            self.entity_type_embed = nn.Embedding(7, params['entity_type_dim'])
        for k in self.edg:
            self.dims[k] = 2*params['out_dim']
        if params['types']:
            for k in self.edg:
                self.dims[k] += 2*params['type_dim']
            self.type_embed = nn.Embedding(3, params['type_dim'])
        temp = int(0.5*self.dims['MM'])
        if params['dist']:
            self.dims['MM'] += params['dist_dim']
            self.dims['SS'] += params['dist_dim']
            self.dist_embed = nn.Embedding(sizes['dist_size'] + 1, 10, padding_idx=sizes['dist_size'])
        if params['gcn']:  # normal_LSTM 110
            self.nodes_change = GCN(temp, temp, params['drop_node'])
        elif params['gat']:
            self.nodes_change = GAT(temp, temp, 2, 0.2, self.device, params['drop_node'])
        elif params['dggcn']:
            self.nodes_change = GraphConvLayer(temp, 2, params['drop_node'], self.device)
        elif params['dggat']:
            self.nodes_change = GraphAttentionLayer(temp, 2, params['drop_node'], 0.2, self.device)
        elif params['multi_gcn']:
            self.nodes_change = MultiGraphConvLayer(temp, 2, params['drop_node'], 2, self.device)
        elif params['multi_gat']:
            self.nodes_change = MultiGraphAttentionLayer(temp, 2, params['drop_node'], 2, 0.2, self.device) 
        self.reduce = nn.ModuleDict()
        for k in self.edg:
            if k != 'EE':
                self.reduce.update({k: nn.Linear(self.dims[k], params['out_dim'], bias=False)})
            elif (('EE' in params['edges']) or ('FULL' in params['edges'])) and (k == 'EE'):
                self.ee = True
                self.reduce.update({k: nn.Linear(self.dims[k], params['out_dim'], bias=False)})
            else:
                self.ee = False
        if params['single']:
            self.walk = WalkLayer(input_size=params['out_dim'],
                                  iters=params['walks_iter'],
                                  beta=params['beta'],
                                  device=self.device)
        else:
            self.walk = nn.ModuleDict()
            for k in range(params['walks_iter']):
                self.walk.update({str(k): WalkLayer(input_size=params['out_dim'],
                                                    beta=params['beta'],
                                                    device=self.device)})
        self.classifier = Classifier(in_size=params['out_dim'],
                                     out_size=sizes['rel_size'],
                                     dropout=params['drop_o'])
        self.loss = nn.CrossEntropyLoss()
        # hyper-parameters for tuning
        self.beta = params['beta']
        self.gradc = params['gc']
        self.learn = params['lr']
        self.reg = params['reg']
        self.out_dim = params['out_dim']
        self.mappings = {'type': maps['type2idx'], 'dist': maps['dist2idx']}
        self.inv_mappings = {'type': maps['idx2type'], 'dist': maps['idx2dist']}
        self.word_dim = params['word_dim']
        self.walks_iter = params['walks_iter']
        self.rel_size = sizes['rel_size']
        self.types = params['types']
        self.ignore_label = lab2ign
        self.dist = params['dist']
        self.gcn_has = params['gcn']
        self.gat_has = params['gat']
        self.dggcn_has = params['dggcn']
        self.dggat_has = params['dggat']
        self.multi_gcn_has = params['multi_gcn']
        self.multi_gat_has = params['multi_gat']
        self.activate = params['activate']
        self.sent = params['sent']
        self.single = params['single']
