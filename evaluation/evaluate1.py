#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/05/2019

author: fenia
"""

import argparse


def prf(tp, fp, fn):
    micro_p = float(tp) / (tp + fp) if (tp + fp != 0) else 0.0
    micro_r = float(tp) / (tp + fn) if (tp + fn != 0) else 0.0
    micro_f = ((2 * micro_p * micro_r) / (micro_p + micro_r)) if micro_p != 0.0 and micro_r != 0.0 else 0.0
    return [micro_p, micro_r, micro_f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=str)
    parser.add_argument('--pred', type=str)
    parser.add_argument('--label', type=str)
    args = parser.parse_args()

    with open(args.pred) as pred, open(args.gold) as gold:
        preds0 = []
        preds1 = []
        preds2 = []
        preds3 = []
        preds4 = []
        preds55 = []
        gold0 = []
        gold1 = []
        gold2 = []
        gold3 = []
        gold4 = []
        gold55 = []

        preds_all = []
        preds_intra = []
        preds_inter = []

        golds_all = []
        golds_intra = []
        golds_inter = []

        for line in pred:
            line = line.rstrip().split('|')
            if line[5] == args.label:
                if (line[0], line[1], line[2], line[3], line[5]) not in preds_all:
                    preds_all += [(line[0], line[1], line[2], line[3], line[5])]
                if ((line[0], line[1], line[2], line[5]) not in preds_inter) and (line[3] == 'CROSS'):
                    preds_inter += [(line[0], line[1], line[2], line[5])]
                if ((line[0], line[1], line[2], line[5]) not in preds_intra) and (line[3] == 'NON-CROSS'):
                    preds_intra += [(line[0], line[1], line[2], line[5])]
                # if ((line[0], line[1], line[2], line[5]) not in preds0) and (line[4] == '0'):
                #     preds0 += [(line[0], line[1], line[2], line[5])]
                # if ((line[0], line[1], line[2], line[5]) not in preds1) and (line[4] == '1'):
                #     preds1 += [(line[0], line[1], line[2], line[5])]
                # if ((line[0], line[1], line[2], line[5]) not in preds2) and (line[4] == '2'):
                #     preds2 += [(line[0], line[1], line[2], line[5])]
                # if ((line[0], line[1], line[2], line[5]) not in preds3) and (line[4] == '3'):
                #     preds3 += [(line[0], line[1], line[2], line[5])]
                # if ((line[0], line[1], line[2], line[5]) not in preds3) and (line[4] == '4'):
                #     preds4 += [(line[0], line[1], line[2], line[5])]
                # if ((line[0], line[1], line[2], line[5]) not in preds3) and (int(line[4]) >= 5):
                #     preds55 += [(line[0], line[1], line[2], line[5])]

        for line2 in gold:
            line2 = line2.rstrip().split('|')
            if line2[5] == args.label:

                if (line2[0], line2[1], line2[2], line2[3], line2[5]) not in golds_all:
                    golds_all += [(line2[0], line2[1], line2[2], line2[3], line2[5])]
                
                if ((line2[0], line2[1], line2[2], line2[5]) not in golds_inter) and (line2[3] == 'CROSS'):
                    golds_inter += [(line2[0], line2[1], line2[2], line2[5])]
                
                if ((line2[0], line2[1], line2[2], line2[5]) not in golds_intra) and (line2[3] == 'NON-CROSS'):
                    golds_intra += [(line2[0], line2[1], line2[2], line2[5])]
                # if ((line2[0], line2[1], line2[2], line2[5]) not in gold0) and (line2[4] == '0'):
                #     gold0 += [(line2[0], line2[1], line2[2], line2[5])]
                # if ((line2[0], line2[1], line2[2], line2[5]) not in gold1) and (line2[4] == '1'):
                #     gold1 += [(line2[0], line2[1], line2[2], line2[5])]
                # if ((line2[0], line2[1], line2[2], line2[5]) not in gold2) and (line2[4] == '2'):
                #     gold2 += [(line2[0], line2[1], line2[2], line2[5])]
                # if ((line2[0], line2[1], line2[2], line2[5]) not in gold3) and (line2[4] == '3'):
                #     gold3 += [(line2[0], line2[1], line2[2], line2[5])]
                # if ((line2[0], line2[1], line2[2], line2[5]) not in gold4) and (line2[4] == '4'):
                #     gold4 += [(line2[0], line2[1], line2[2], line2[5])]
                # if ((line2[0], line2[1], line2[2], line2[5]) not in gold55) and (int(line2[4]) >= 5):
                #     gold55 += [(line2[0], line2[1], line2[2], line2[5])]

        tp = len([a for a in preds_all if a in golds_all])
        tp_intra = len([a for a in preds_intra if a in golds_intra])
        tp_inter = len([a for a in preds_inter if a in golds_inter])
        # tp0 = len([a for a in preds0 if a in gold0])
        # tp1 = len([a for a in preds1 if a in gold1])
        # tp2 = len([a for a in preds2 if a in gold2])
        # tp3 = len([a for a in preds3 if a in gold3])
        # tp4 = len([a for a in preds4 if a in gold4])
        # tp5 = len([a for a in preds55 if a in gold55])
        # fp0 = len([a for a in preds0 if a not in gold0])
        # fp1 = len([a for a in preds1 if a not in gold1])
        # fp2 = len([a for a in preds2 if a not in gold2])
        # fp3 = len([a for a in preds3 if a not in gold3])
        # fp4 = len([a for a in preds4 if a not in gold4])
        # fp5 = len([a for a in preds55 if a not in gold55])
        # fn0 = len([a for a in gold0 if a not in preds0])
        # fn1 = len([a for a in gold1 if a not in preds1])
        # fn2 = len([a for a in gold2 if a not in preds2])
        # fn3 = len([a for a in gold3 if a not in preds3])
        # fn4 = len([a for a in gold4 if a not in preds4])
        # fn5 = len([a for a in gold55 if a not in preds55])

        fp = len([a for a in preds_all if a not in golds_all])
        fp_intra = len([a for a in preds_intra if a not in golds_intra])
        fp_inter = len([a for a in preds_inter if a not in golds_inter])
        
        fn = len([a for a in golds_all if a not in preds_all])
        fn_intra = len([a for a in golds_intra if a not in preds_intra])
        fn_inter = len([a for a in golds_inter if a not in preds_inter])

        r1 = prf(tp, fp, fn)
        r2 = prf(tp_intra, fp_intra, fn_intra)
        r3 = prf(tp_inter, fp_inter, fn_inter)
        # r0 = prf(tp0, fp0, fn0)
        # r1 = prf(tp1, fp1, fn1)
        # r2 = prf(tp2, fp2, fn2)
        # r3 = prf(tp3, fp3, fn3)
        # r4 = prf(tp4, fp4, fn4)
        # r5 = prf(tp5, fp5, fn5)
        # print('WALK 0 P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r0[0], r0[1], r0[2],
        #                                                                        tp0 + fn0, tp0, fp0, fn0))
        # print('WALK 1 P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r1[0], r1[1], r1[2],
        #                                                                        tp1 + fn1, tp1, fp1, fn1))
        # print('WALK 2 P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r2[0], r2[1], r2[2],
        #                                                                        tp2 + fn2, tp2, fp2, fn2))
        # print('WALK 3 P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r3[0], r3[1], r3[2],
        #                                                                        tp3 + fn3, tp3, fp3, fn3))
        # print('WALK 4 P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r4[0], r4[1], r4[2],
        #                                                                        tp4 + fn4, tp4, fp4, fn4))
        # print('WALK 5 P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r5[0], r5[1], r5[2],
        #                                                                        tp5 + fn5, tp5, fp5, fn5))

        print('                                          TOTAL\tTP\tFP\tFN')
        print('Overall P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r1[0], r1[1], r1[2],
                                                                                tp + fn, tp, fp, fn))
        print('INTRA P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r2[0], r2[1], r2[2],
                                                                              tp_intra + fn_intra,
                                                                              tp_intra, fp_intra,
                                                                              fn_intra))
        print('INTER P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r3[0], r3[1], r3[2],
                                                                              tp_inter + fn_inter,
                                                                              tp_inter, fp_inter,
                                                                              fn_inter))


# python evaluate.py --pred ../results/cdr/01.dcgat/test.preds --gold ../data/CDR/processed/test_filter.gold --label 1:CID:2

if __name__ == "__main__":
    main()
