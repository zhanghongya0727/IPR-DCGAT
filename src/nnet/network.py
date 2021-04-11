#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, Fenia Christopoulou, National Centre for Text Mining,
# School of Computer Science, The University of Manchester.
# https://github.com/fenchri/edge-oriented-graph/

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from nnet.init_net import BaseNet


class EOG(BaseNet):

    def input_layer(self, words_, positions_, entity_type_):  # GDA dataset
        word_vec = self.word_embed(words_)
        position_vec = self.conference_embed(positions_)
        entity_type_vec = self.entity_type_embed(entity_type_)
        word_vec = torch.cat([word_vec, position_vec, entity_type_vec], dim=-1)  # [word_num * dim]
        return word_vec

    def encoding_layer(self, word_vec, word_sec):
        ys = self.encoder(torch.split(word_vec, word_sec.tolist(), dim=0), word_sec)
        return ys

    def input_layer_new(self, doc_sec, word_sec, words_, positions_, entity_type_):  # 新的预处理方式
        one = torch.ones(1)
        word_vec = self.word_embed(words_)
        position_vec = self.conference_embed(positions_)
        entity_type_vec = self.entity_type_embed(entity_type_)
        word_vec = torch.cat([word_vec, position_vec, entity_type_vec], dim=-1)  # [word_num * dim]
        word_sec = torch.tensor([word_sec[: id + 1].sum() for id in range(word_sec.size(0))])
        doc_temp = torch.tensor([doc_sec[: id + 1].sum() for id in range(doc_sec.size(0))])
        context_seg = torch.zeros(word_vec.size(0))
        context_seg[0] = one
        temp_ = word_sec - 1  # 正确的
        context_seg[temp_] = one
        context_seg[doc_temp[: doc_sec.size(0) - 1]] = one
        # 把word_vec, context_seg按照文档划分而不是batch之内所有的文档串联
        word_vec = torch.split(word_vec, doc_sec.tolist(), dim=0)
        word_vec = pad_sequence(word_vec, batch_first=True, padding_value=0)
        context_seg = torch.split(context_seg, doc_sec.tolist(), dim=0)
        context_seg = pad_sequence(context_seg, batch_first=True, padding_value=0)
        return word_vec, context_seg

    def doc_encoder(self, input_sent, context_seg, doc_sec, word_sec):
        batch_size = context_seg.shape[0]
        docs_emb = []  # sentence embedding
        docs_len = []
        sents_emb = []
        for batch_no in range(batch_size):
            sent_list = []
            sent_lens = []
            sent_index = ((context_seg[batch_no] == 1).nonzero()).squeeze(-1).tolist()
            pre_index = 0
            for i, index in enumerate(sent_index):
                if i != 0:
                    if i == 1:
                        sent_list.append(input_sent[batch_no][pre_index:index + 1])
                        sent_lens.append(index - pre_index + 1)
                    else:
                        sent_list.append(input_sent[batch_no][pre_index + 1:index + 1])
                        sent_lens.append(index - pre_index)
                pre_index = index
            sents = pad_sequence(sent_list).permute(1, 0, 2)
            sent_lens_t = torch.LongTensor(sent_lens).cuda()
            docs_len.append(sent_lens)
            sents_output, sent_emb = self.rnn_sent(sents, sent_lens_t)  # sentence embeddings for a document.
            doc_emb = None
            for i, (sen_len, emb) in enumerate(zip(sent_lens, sents_output)):
                if i == 0:
                    doc_emb = emb[:sen_len]
                else:
                    doc_emb = torch.cat([doc_emb, emb[:sen_len]], dim=0)
            docs_emb.append(doc_emb)
            sents_emb.append(sent_emb.squeeze(1))
        docs_emb = pad_sequence(docs_emb).permute(1, 0, 2)  # B * # sentence * Dimention
        docs_emb = self.dropout_rate(torch.relu(self.linear_re(docs_emb)))
        docs_emb = torch.cat([docs_emb[i][: doc_sec[i]] for i in range(doc_sec.size(0))])
        docs_emb = torch.split(docs_emb, word_sec.tolist(), dim=0)
        docs_emb = pad_sequence(docs_emb, batch_first=True, padding_value=0)
        return docs_emb

    def node_layer(self, encoded_seq, info, word_sec, section, sent):
        if sent == 'mean':
            sentences = torch.mean(encoded_seq, dim=1)
        elif sent == 'max':
            sentences = torch.max(encoded_seq, dim=1)[0]
        elif sent == 'att':
            sentences = self.sents(encoded_seq)
        elif sent == 'p_means':
            maxx = torch.max(encoded_seq, dim=1)[0]
            minn = torch.min(encoded_seq, dim=1)[0]
            mean = torch.mean(encoded_seq, dim=1)
            sentences = torch.cat((mean, maxx, minn), dim=1)
            sentences = self.reduce['reduce_sent'](sentences)
        else:
            print('no sentences encoder')
        temp_ = torch.arange(word_sec.max()).unsqueeze(0).repeat(sentences.size(0), 1).to(self.device)
        remove_pad = (temp_ < word_sec.unsqueeze(1))
        mentions = self.merge_tokens(info, encoded_seq, remove_pad)  # mention nodes
        entities = self.merge_mentions(info, mentions)  # entity nodes
        nodes = torch.cat((entities, mentions, sentences), dim=0)  # e + m + s (all)
        nodes_info = self.node_info(section, info)  # info/node: node type | semantic type | sentence ID
        if self.types:  # + node types
            nodes = torch.cat((nodes, self.type_embed(nodes_info[:, 0])), dim=1)
        nodes = self.rearrange_nodes(nodes, section)
        nodes = self.split_n_pad(nodes, section, pad=0)
        nodes_info = self.rearrange_nodes(nodes_info, section)
        nodes_info = self.split_n_pad(nodes_info, section, pad=-1)
        return nodes, nodes_info, mentions

    def edge_layer(self, nodes, nodes_info, section, positions):
        r_idx, c_idx = torch.meshgrid(torch.arange(nodes.size(1)).to(self.device),
                                      torch.arange(nodes.size(1)).to(self.device))
        graph = torch.cat((nodes[:, r_idx], nodes[:, c_idx]), dim=3)
        r_id, c_id = nodes_info[..., 0][:, r_idx], nodes_info[..., 0][:, c_idx]  # node type indicators
        pid = self.pair_ids(r_id, c_id)
        reduced_graph = torch.where(pid['MS'].unsqueeze(-1), self.reduce['MS'](graph),
                                    torch.zeros(graph.size()[:-1] + (self.out_dim,)).to(self.device))
        reduced_graph = torch.where(pid['ME'].unsqueeze(-1), self.reduce['ME'](graph), reduced_graph)
        reduced_graph = torch.where(pid['ES'].unsqueeze(-1), self.reduce['ES'](graph), reduced_graph)
        dist_vec = self.dist_embed(positions)  # distances
        if self.dist:
            reduced_graph = torch.where(pid['SS'].unsqueeze(-1),
                                        self.reduce['SS'](torch.cat((graph, dist_vec), dim=3)), reduced_graph)
        else:
            reduced_graph = torch.where(pid['SS'].unsqueeze(-1), self.reduce['SS'](graph), reduced_graph)
        if self.dist:
            reduced_graph = torch.where(pid['MM'].unsqueeze(-1),
                                        self.reduce['MM'](torch.cat((graph, dist_vec), dim=3)), reduced_graph)
        else:
            reduced_graph = torch.where(pid['MM'].unsqueeze(-1), self.reduce['MM'](graph), reduced_graph)
        if self.ee:
            reduced_graph = torch.where(pid['EE'].unsqueeze(-1), self.reduce['EE'](graph), reduced_graph)
        mask = self.get_nodes_mask(section.sum(dim=1))
        return reduced_graph, (r_idx, c_idx), nodes_info, mask

    @staticmethod
    def pair_ids(r_id, c_id):
        pids = {
            'EE': ((r_id == 0) & (c_id == 0)),
            'MM': ((r_id == 1) & (c_id == 1)),
            'SS': ((r_id == 2) & (c_id == 2)),
            'ES': (((r_id == 0) & (c_id == 2)) | ((r_id == 2) & (c_id == 0))),
            'MS': (((r_id == 1) & (c_id == 2)) | ((r_id == 2) & (c_id == 1))),
            'ME': (((r_id == 1) & (c_id == 0)) | ((r_id == 0) & (c_id == 1)))
        }
        return pids

    @staticmethod
    def rearrange_nodes(nodes, section):
        tmp1 = section.t().contiguous().view(-1).long().to(nodes.device)
        tmp3 = torch.arange(section.numel()).view(section.size(1),
                                                  section.size(0)).t().contiguous().view(-1).long().to(nodes.device)
        tmp2 = torch.arange(section.sum()).to(nodes.device).split(tmp1.tolist())
        tmp2 = pad_sequence(tmp2, batch_first=True, padding_value=-1)[tmp3].view(-1)
        tmp2 = tmp2[(tmp2 != -1).nonzero().squeeze()]  # remove -1 (padded)
        nodes = torch.index_select(nodes, 0, tmp2)
        return nodes

    @staticmethod
    def split_n_pad(nodes, section, pad=None):
        nodes = torch.split(nodes, section.sum(dim=1).tolist())
        nodes = pad_sequence(nodes, batch_first=True, padding_value=pad)
        return nodes

    @staticmethod
    def get_nodes_mask(nodes_size):
        n_total = torch.arange(nodes_size.max()).to(nodes_size.device)
        idx_r, idx_c, idx_d = torch.meshgrid(n_total, n_total, n_total)
        ns = nodes_size[:, None, None, None]
        mask3d = ~(torch.ge(idx_r, ns) | torch.ge(idx_c, ns) | torch.ge(idx_d, ns))
        return mask3d

    def node_info(self, section, info):
        typ = torch.repeat_interleave(torch.arange(3).to(self.device), section.sum(dim=0))  # node types (0,1,2)
        rows_ = torch.bincount(info[:, 0]).cumsum(dim=0).sub(1)
        stypes = torch.neg(torch.ones(section[:, 2].sum())).to(self.device).long()  # semantic type sentences = -1
        all_types = torch.cat((info[:, 1][rows_], info[:, 1], stypes), dim=0)
        sents_ = torch.arange(section.sum(dim=0)[2]).to(self.device)
        sent_id = torch.cat((info[:, 4][rows_], info[:, 4], sents_), dim=0)  # sent_id
        return torch.cat((typ.unsqueeze(-1), all_types.unsqueeze(-1), sent_id.unsqueeze(-1)), dim=1)

    def estimate_loss(self, pred_pairs, truth):
        mask = torch.ne(truth, -1)
        truth = truth[mask]
        pred_pairs = pred_pairs[mask]
        assert (truth != -1).all()
        loss = self.loss(pred_pairs, truth)
        predictions = F.softmax(pred_pairs, dim=1).data.argmax(dim=1)
        # predictions = truth
        stats = self.count_predictions(predictions, truth)
        return loss, stats, predictions

    @staticmethod
    def merge_mentions(info, mentions):
        m_ids, e_ids = torch.broadcast_tensors(info[:, 0].unsqueeze(0),
                                               torch.arange(0, max(info[:, 0]) + 1).unsqueeze(-1).to(info.device))
        index_m = torch.eq(m_ids, e_ids).type('torch.FloatTensor').to(info.device)
        entities = torch.div(torch.matmul(index_m, mentions), torch.sum(index_m, dim=1).unsqueeze(-1))  # average
        return entities

    @staticmethod
    def merge_tokens(info, enc_seq, rm_pad):
        enc_seq = enc_seq[rm_pad]
        start, end, w_ids = torch.broadcast_tensors(info[:, 2].unsqueeze(-1),
                                                    info[:, 3].unsqueeze(-1),
                                                    torch.arange(0, enc_seq.shape[0]).unsqueeze(0).to(info.device))
        index_t = (torch.ge(w_ids, start) & torch.lt(w_ids, end)).float().to(info.device)
        mentions = torch.div(torch.matmul(index_t, enc_seq), torch.sum(index_t, dim=1).unsqueeze(-1))  # average
        return mentions

    def adj_normalize(self, adj_):
        adj_ = adj_.float() + torch.eye(adj_.size(1)).repeat(adj_.size(0), 1, 1).to(self.device)
        rowsum = adj_.sum(2).float()
        r_inv_sqrt = torch.pow(rowsum, -0.5)
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        temp1 = []
        for a in range(r_inv_sqrt.size(0)):
            temp1.append(torch.diag(r_inv_sqrt[a]).unsqueeze(0))
        r_mat_inv_sqrt = torch.cat(temp1)
        adj_ = torch.matmul(adj_, r_mat_inv_sqrt)
        adj_ = adj_.permute(0, 2, 1)
        adj_ = torch.matmul(adj_, r_mat_inv_sqrt)  # d^-0.5*a*d^-0.5
        return adj_

    @staticmethod
    def select_pairs(combs, nodes_info, idx):
        combs = torch.split(combs, 2, dim=0)
        sel = torch.zeros(nodes_info.size(0), nodes_info.size(1), nodes_info.size(1)).to(nodes_info.device)
        a_ = nodes_info[..., 0][:, idx[0]]
        b_ = nodes_info[..., 0][:, idx[1]]
        c_ = nodes_info[..., 1][:, idx[0]]
        d_ = nodes_info[..., 1][:, idx[1]]
        for ca, cb in combs:
            condition1 = torch.eq(a_, 0) & torch.eq(b_, 0)  # needs to be an entity node (id=0)
            condition2 = torch.eq(c_, ca) & torch.eq(d_, cb)  # valid pair semantic types
            sel = torch.where(condition1 & condition2, torch.ones_like(sel), sel)
        return sel.nonzero().unbind(dim=1)

    def count_predictions(self, y, t):
        label_num = torch.as_tensor([self.rel_size]).long().to(self.device)
        ignore_label = torch.as_tensor([self.ignore_label]).long().to(self.device)
        mask_t = torch.eq(t, ignore_label).view(-1)  # where the ground truth needs to be ignored
        mask_p = torch.eq(y, ignore_label).view(-1)  # where the predicted needs to be ignored
        true = torch.where(mask_t, label_num, t.view(-1).long().to(self.device))  # ground truth
        pred = torch.where(mask_p, label_num, y.view(-1).long().to(self.device))  # output of NN
        tp_mask = torch.where(torch.eq(pred, true), true, label_num)
        fp_mask = torch.where(torch.ne(pred, true), pred, label_num)
        fn_mask = torch.where(torch.ne(pred, true), true, label_num)
        tp = torch.bincount(tp_mask, minlength=self.rel_size + 1)[:self.rel_size]
        fp = torch.bincount(fp_mask, minlength=self.rel_size + 1)[:self.rel_size]
        fn = torch.bincount(fn_mask, minlength=self.rel_size + 1)[:self.rel_size]
        tn = torch.sum(mask_t & mask_p)
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def forward(self, batch):
        # word_vec = self.input_layer(batch['words'], batch['position'], batch['entity_type'])  # 旧版
        # encoded_seq = self.encoding_layer(word_vec, batch['word_sec'])
        word_vec, context_seg = self.input_layer_new(batch['doc_sec'], batch['word_sec'],
                                                     batch['words'], batch['position'], batch['entity_type'])
        encoded_seq = self.doc_encoder(word_vec, context_seg, batch['doc_sec'], batch['word_sec'])
        nodes, nodes_info, mentions = self.node_layer(encoded_seq, batch['entities'], batch['word_sec'],
                                                      batch['section'], 'mean')
        if self.gcn_has or self.gat_has or self.multi_gcn_has or self.multi_gat_has or self.dggcn_has or self.dggat_has:
            adj_ = self.adj_normalize(adj_=batch['adjacency'])
            nodes = self.nodes_change(nodes, adj_)
        graph, pindex, nodes_info, mask = self.edge_layer(nodes, nodes_info, batch['section'], batch['distances'])
        graph = torch.where(batch['adjacency'].unsqueeze(-1), graph, torch.zeros_like(graph))
        if self.single:
            if self.walks_iter and self.walks_iter > 0:
                for _ in range(self.walks_iter):
                    graph = self.walk(graph, mask_=mask, activate='sigmoid')
        else:
            if self.walks_iter and self.walks_iter > 0:  # different w
                for a in range(self.walks_iter):
                    graph = self.walk[str(a)](graph, mask_=mask, activate=self.activate)
        select = self.select_pairs(batch['pairs4class'], nodes_info, pindex)
        graph = self.classifier(graph[select])
        loss, stats, preds = self.estimate_loss(graph, batch['relations'][select].long())
        return loss, stats, preds, select
