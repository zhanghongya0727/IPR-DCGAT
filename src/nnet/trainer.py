#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, Fenia Christopoulou, National Centre for Text Mining,
# School of Computer Science, The University of Manchester.
# https://github.com/fenchri/edge-oriented-graph/

import torch
import numpy as np
import os
from time import time
import itertools
import copy
import datetime
import random
from random import shuffle
from utils import print_results, write_preds, write_errors, print_options
from converter import concat_examples
from torch import autograd
from nnet.network import EOG
from torch import nn, optim
import sys
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)


class Trainer:
    def __init__(self, loader, params, data, model_folder):
        self.data = data
        self.params = params
        self.rel_size = loader.n_rel
        self.loader = loader
        self.model_folder = model_folder
        self.device = torch.device("cuda:{}".format(params['gpu']) if params['gpu'] != -1 else "cpu")
        self.gc = params['gc']
        self.epoch = params['epoch']
        self.example = params['example']
        self.pa = params['param_avg']
        self.es = params['early_stop']
        self.primary_metric = params['primary_metric']
        self.show_class = params['show_class']
        self.preds_file = os.path.join(model_folder, params['save_pred'])
        self.best_epoch = 0
        self.train_res = {'loss': [], 'score': []}
        self.test_res = {'loss': [], 'score': []}
        self.max_patience = self.params['patience']
        self.cur_patience = 0
        self.best_score = 0.0
        self.best_epoch = 0
        self.pairs4train = []
        for i in params['include_pairs']:
            m, n = i.split('-')
            self.pairs4train += [self.loader.type2index[m], self.loader.type2index[n]]
        self.pairs4class = []
        for i in params['classify_pairs']:
            m, n = i.split('-')
            self.pairs4class += [self.loader.type2index[m], self.loader.type2index[n]]
        if params['param_avg']:
            self.averaged_params = {}
        self.model = self.init_model()
        self.optimizer = self.set_optimizer(self.model)

    def init_model(self):
        model_0 = EOG(self.params,
                      sizes={  # 'word_size': self.loader.n_words,  # 如果是GDA要取消注释
                          'dist_size': self.loader.n_dist,
                          'type_size': self.loader.n_type, 'rel_size': self.loader.n_rel},
                      maps={  # 'word2idx': self.loader.word2index, 'idx2word': self.loader.index2word, # 如果是GDA要取消注释
                          'rel2idx': self.loader.rel2index, 'idx2rel': self.loader.index2rel,
                          'type2idx': self.loader.type2index, 'idx2type': self.loader.index2type,
                          'dist2idx': self.loader.dist2index, 'idx2dist': self.loader.index2dist},
                      lab2ign=self.loader.label2ignore)
        if self.params['gpu'] != -1:
            torch.cuda.set_device(self.device)
            model_0.to(self.device)
        return model_0

    def set_optimizer(self, model_0):
        params2reg = []
        params0reg = []
        for p_name, p_value in model_0.named_parameters():
            if '.bias' in p_name:
                params0reg += [p_value]
            else:
                params2reg += [p_value]
        assert len(params0reg) + len(params2reg) == len(list(model_0.parameters()))
        groups = [dict(params=params2reg), dict(params=params0reg, weight_decay=.0)]
        optimizer = optim.Adam(groups, lr=self.params['lr'], weight_decay=self.params['reg'], amsgrad=True)
        print_options(self.params)
        for p_name, p_value in model_0.named_parameters():
            if p_value.requires_grad:
                print(p_name)
        return optimizer

    @staticmethod
    def iterator(x, shuffle_=False, batch_size=1):
        if shuffle_:
            shuffle(x)
        new = [x[i:i + batch_size] for i in range(0, len(x), batch_size)]
        return new

    def run(self):
        print('\n======== START TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))
        random.shuffle(self.data['train'])  # shuffle training data at least once
        for epoch in range(1, self.epoch + 1):
            self.train_epoch(epoch)
            if self.pa:
                self.parameter_averaging()
            self.eval_epoch()
            stop = self.epoch_checking(epoch)
            if stop and self.es:
                break
            if self.pa:
                self.parameter_averaging(reset=True)
        print('Best epoch: {}'.format(self.best_epoch))
        if self.pa:
            self.parameter_averaging(epoch=self.best_epoch)
        self.eval_epoch(final=True, save_predictions=True)
        print('\n======== END TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))

    def train_epoch(self, epoch):
        t1 = time()
        output = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'loss': [], 'preds': []}
        train_info = []
        self.model = self.model.train()
        train_iter = self.iterator(self.data['train'], batch_size=self.params['batch'],
                                   shuffle_=self.params['shuffle_data'])
        for batch_idx, batch in enumerate(train_iter):
            batch = self.convert_batch(batch)
            with autograd.detect_anomaly():
                self.optimizer.zero_grad()
                loss, stats, predictions, select = self.model(batch)
                loss.backward()  # backward computation
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gc)  # gradient clipping
                self.optimizer.step()  # update
                print("step :{:02d} is success ".format(batch_idx), end="\r")
            output['loss'] += [loss.item()]
            output['tp'] += [stats['tp'].to('cpu').data.numpy()]
            output['fp'] += [stats['fp'].to('cpu').data.numpy()]
            output['fn'] += [stats['fn'].to('cpu').data.numpy()]
            output['tn'] += [stats['tn'].to('cpu').data.numpy()]
            output['preds'] += [predictions.to('cpu').data.numpy()]
            train_info += [batch['info'][select[0].to('cpu').data.numpy(),
                                         select[1].to('cpu').data.numpy(),
                                         select[2].to('cpu').data.numpy()]]
            torch.cuda.empty_cache()
        t2 = time()
        total_loss, scores = self.performance(output)
        self.train_res['loss'] += [total_loss]
        self.train_res['score'] += [scores[self.primary_metric]]
        print('Epoch: {:02d} | TRAIN | LOSS = {:.05f}, '.format(epoch, total_loss), end="")
        print_results(scores, t2 - t1)

    def eval_epoch(self, final=False, save_predictions=False):
        t1 = time()
        output = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'loss': [], 'preds': []}
        test_info = []
        self.model = self.model.eval()
        test_iter = self.iterator(self.data['test'], batch_size=self.params['batch'], shuffle_=False)
        for batch_idx, batch in enumerate(test_iter):
            batch = self.convert_batch(batch)
            with torch.no_grad():
                loss, stats, predictions, select = self.model(batch)
                output['loss'] += [loss.item()]
                output['tp'] += [stats['tp'].to('cpu').data.numpy()]
                output['fp'] += [stats['fp'].to('cpu').data.numpy()]
                output['fn'] += [stats['fn'].to('cpu').data.numpy()]
                output['tn'] += [stats['tn'].to('cpu').data.numpy()]
                output['preds'] += [predictions.to('cpu').data.numpy()]
                test_info += [batch['info'][select[0].to('cpu').data.numpy(),
                                            select[1].to('cpu').data.numpy(),
                                            select[2].to('cpu').data.numpy()]]
        t2 = time()
        total_loss, scores = self.performance(output)
        if not final:
            self.test_res['loss'] += [total_loss]
            self.test_res['score'] += [scores[self.primary_metric]]
        print('            TEST  | LOSS = {:.05f}, '.format(total_loss), end="")
        print_results(scores, t2 - t1)
        print()
        if save_predictions:
            write_preds(output['preds'], test_info, self.preds_file, map_=self.loader.index2rel)
            write_errors(output['preds'], test_info, self.preds_file, map_=self.loader.index2rel)

    def parameter_averaging(self, epoch=None, reset=False):
        for p_name, p_value in self.model.named_parameters():
            if p_name not in self.averaged_params:
                self.averaged_params[p_name] = []
            if reset:
                p_new = copy.deepcopy(self.averaged_params[p_name][-1])  # use last epoch param
            elif epoch:
                p_new = np.mean(self.averaged_params[p_name][:epoch], axis=0)  # estimate average until this epoch
            else:
                self.averaged_params[p_name].append(p_value.data.to('cpu').numpy())
                p_new = np.mean(self.averaged_params[p_name], axis=0)  # estimate average
            if self.device != 'cpu':
                p_value.data = torch.from_numpy(p_new).to(self.device)
            else:
                p_value.data = torch.from_numpy(p_new)

    def epoch_checking(self, epoch):
        if self.test_res['score'][-1] > self.best_score:  # improvement
            self.best_score = self.test_res['score'][-1]
            self.cur_patience = 0
            if self.es:
                self.best_epoch = epoch
        else:
            self.cur_patience += 1
            if not self.es:
                self.best_epoch = epoch
        if epoch % 5 == 0 and self.es:
            print('Current best {} score {:.6f} @ epoch {}\n'.format(self.params['primary_metric'],
                                                                     self.best_score, self.best_epoch))
        if self.max_patience == self.cur_patience and self.es:  # early stop must happen
            self.best_epoch = epoch - self.max_patience
            return True
        else:
            return False

    @staticmethod
    def performance(stats):
        def fbeta_score(precision, recall, beta=1.0):
            beta_square = beta * beta
            if (precision != 0.0) and (recall != 0.0):
                res = ((1 + beta_square) * precision * recall / (beta_square * precision + recall))
            else:
                res = 0.0
            return res

        def prf1(tp_, fp_, fn_, tn_):
            tp_ = np.sum(tp_, axis=0)
            fp_ = np.sum(fp_, axis=0)
            fn_ = np.sum(fn_, axis=0)
            tn_ = np.sum(tn_, axis=0)
            atp = np.sum(tp_)
            afp = np.sum(fp_)
            afn = np.sum(fn_)
            atn = np.sum(tn_)
            micro_p = (1.0 * atp) / (atp + afp) if (atp + afp != 0) else 0.0
            micro_r = (1.0 * atp) / (atp + afn) if (atp + afn != 0) else 0.0
            micro_f = fbeta_score(micro_p, micro_r)
            pp = [0]
            rr = [0]
            ff = [0]
            macro_p = np.mean(pp)
            macro_r = np.mean(rr)
            macro_f = np.mean(ff)
            acc = (atp + atn) / (atp + atn + afp + afn) if (atp + atn + afp + afn) else 0.0
            return {'acc': acc,
                    'micro_p': micro_p, 'micro_r': micro_r, 'micro_f': micro_f,
                    'macro_p': macro_p, 'macro_r': macro_r, 'macro_f': macro_f,
                    'tp': atp, 'true': atp + afn, 'pred': atp + afp, 'total': (atp + atn + afp + afn)}

        fin_loss = sum(stats['loss']) / len(stats['loss'])
        scores = prf1(stats['tp'], stats['fp'], stats['fn'], stats['tn'])
        return fin_loss, scores

    def convert_batch(self, batch):
        new_batch = {'entities': []}
        ent_count, sent_count, word_count = 0, 0, 0
        full_text = []

        # TODO make this faster
        for i, b in enumerate(batch):
            current_text = list(itertools.chain.from_iterable(b['text']))
            full_text += current_text

            temp = []
            for e in b['ents']:
                # token ids are correct
                assert full_text[(e[2] + word_count):(e[3] + word_count)] == current_text[e[2]:e[3]], \
                    '{} != {}'.format(full_text[(e[2] + word_count):(e[3] + word_count)], current_text[e[2]:e[3]])
                temp += [[e[0] + ent_count, e[1], e[2] + word_count, e[3] + word_count, e[4] + sent_count]]

            new_batch['entities'] += [np.array(temp)]
            word_count += sum([len(s) for s in b['text']])
            ent_count = max([t[0] for t in temp]) + 1
            sent_count += len(b['text'])
        new_batch['entities'] = torch.as_tensor(np.concatenate(new_batch['entities'], axis=0)).long().to(self.device)
        batch_ = [{k: v for k, v in b.items() if (k != 'info' and k != 'text')} for b in batch]
        converted_batch = concat_examples(batch_, device=self.device, padding=-1)
        converted_batch['adjacency'][converted_batch['adjacency'] == -1] = 0
        converted_batch['dist'][converted_batch['dist'] == -1] = self.loader.n_dist
        new_batch['adjacency'] = converted_batch['adjacency'].byte()
        new_batch['distances'] = converted_batch['dist']
        new_batch['relations'] = converted_batch['rels']
        new_batch['section'] = converted_batch['section']
        new_batch['word_sec'] = converted_batch['word_sec'][converted_batch['word_sec'] != -1].long()
        new_batch['doc_sec'] = converted_batch['doc_sec'][converted_batch['doc_sec'] != -1].long()
        new_batch['words'] = converted_batch['words'][converted_batch['words'] != -1].long()
        new_batch['position'] = converted_batch['position'][converted_batch['words'] != -1].long()
        new_batch['entity_type'] = converted_batch['entity_type'][converted_batch['words'] != -1].long()
        new_batch['pairs4class'] = torch.as_tensor(self.pairs4class).long().to(self.device)
        new_batch['info'] = np.stack([np.array(np.pad(b['info'],
                                                      ((0,
                                                        new_batch['section'][:, 0].sum(dim=0).item() - b['info'].shape[
                                                            0]),
                                                       (0,
                                                        new_batch['section'][:, 0].sum(dim=0).item() - b['info'].shape[
                                                            0])),
                                                      'constant',
                                                      constant_values=(-1, -1))) for b in batch], axis=0)
        return new_batch
