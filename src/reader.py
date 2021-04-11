#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, Fenia Christopoulou, National Centre for Text Mining,
# School of Computer Science, The University of Manchester.
# https://github.com/fenchri/edge-oriented-graph/
from collections import OrderedDict
from recordtype import recordtype

EntityInfo = recordtype('EntityInfo', 'id type mstart mend sentNo')
PairInfo = recordtype('PairInfo', 'type direction cross')


def chunks(l, n):
    """
    Successive n-sized chunks from l.
    """
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


def read(input_file, documents, entities, relations):
    lengths = []
    sents = []
    with open(input_file, 'r') as infile:
        for line in infile:
            line = line.rstrip().split('\t')
            pmid = line[0]
            text = line[1]
            prs = chunks(line[2:], 17)
            if pmid not in documents:
                documents[pmid] = [t.split(' ') for t in text.split('|')]
            if pmid not in entities:
                entities[pmid] = OrderedDict()
            if pmid not in relations:
                relations[pmid] = OrderedDict()
            lengths += [max([len(s) for s in documents[pmid]])]
            sents += [len(text.split('|'))]
            allp = 0
            for p in prs:
                if (p[5], p[11]) not in relations[pmid]:
                    relations[pmid][(p[5], p[11])] = PairInfo(p[0], p[1], p[2])
                    allp += 1
                else:
                    print('duplicates!')
                if p[5] not in entities[pmid]:
                    entities[pmid][p[5]] = EntityInfo(p[5], p[7], p[8], p[9], p[10])

                if p[11] not in entities[pmid]:
                    entities[pmid][p[11]] = EntityInfo(p[11], p[13], p[14], p[15], p[16])
            assert len(relations[pmid]) == allp
            # print(entities[pmid])
    todel = []
    for pmid, d in relations.items():
        if not relations[pmid]:
            todel += [pmid]
    for pmid in todel:
        del documents[pmid]
        del entities[pmid]
        del relations[pmid]

    return lengths, sents, documents, entities, relations
