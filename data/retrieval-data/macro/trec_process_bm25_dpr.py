## This is for generating data for macro1M from trec 2019 & trec 2020

import os
import json
import shutil

import random
import pickle
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser

import torch
import random
import numpy as np

from IPython import embed

from collections import defaultdict

parser = ArgumentParser()
parser.add_argument('--qrel_file', type=str, required=True)
parser.add_argument('--qrel_binary_file', type=str, required=True)
parser.add_argument('--query_file', type=str, required=True)
parser.add_argument('--corpus_file', type=str, required=True)
parser.add_argument('--docid2idx_file', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)

args = parser.parse_args()

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# read corpus
corpus_id2doc = {}
with open(args.corpus_file) as f:
    readin = f.readlines()
    for line in tqdm(readin):
        tmp = line.strip().split('\t')
        corpus_id2doc[tmp[0]] = tmp[1]
docid_set = set(corpus_id2doc.keys())


# read qrel
qrel_dict = {}
with open(args.qrel_file) as f:
    readin = json.load(f)
    for qid in readin:
        tmp_dict = {}
        for docid in readin[qid]:
            if docid in docid_set:
                tmp_dict[docid] = readin[qid][docid]
        total_rel = sum(list(tmp_dict.values()))
        if total_rel > 0:
            qrel_dict[qid] = tmp_dict

# read qrel
qrel_binary_dict = {}
with open(args.qrel_binary_file) as f:
    readin = json.load(f)
    for qid in readin:
        tmp_dict = {}
        for docid in readin[qid]:
            if docid in docid_set:
                tmp_dict[docid] = readin[qid][docid]
        total_rel = sum(list(tmp_dict.values()))
        if total_rel > 0:
            qrel_binary_dict[qid] = tmp_dict

# read query
query_dict = {}
query_binary_dict = {}
with open(args.query_file) as f:
    readin = f.readlines()
    for line in readin:
        tmp = line.strip().split('\t')
        if tmp[0] in qrel_dict:
            query_dict[tmp[0]] = tmp[1]
        if tmp[0] in qrel_binary_dict:
            query_binary_dict[tmp[0]] = tmp[1]


# (bm25) save query & ground truth trec
if not os.path.exists(os.path.join(args.save_dir, 'bm25')):
    os.mkdir(os.path.join(args.save_dir, 'bm25'))


with open(os.path.join(args.save_dir, 'bm25', 'test.query.txt'), 'w') as fout:
    for qid in query_dict:
        fout.write(qid + '\t' + query_dict[qid] + '\n')


with open(os.path.join(args.save_dir, 'bm25', 'test.query.binary.txt'), 'w') as fout:
    for qid in query_binary_dict:
        fout.write(qid + '\t' + query_binary_dict[qid] + '\n')


with open(os.path.join(args.save_dir, 'bm25', 'test.truth.trec'), 'w') as fout:
    for qid in qrel_dict:
        for docid, rel in qrel_dict[qid].items():
            fout.write(qid + ' ' + str(0) + ' ' + str(docid) + ' ' + str(rel) + '\n')


with open(os.path.join(args.save_dir, 'bm25', 'test.truth.binary.trec'), 'w') as fout:
    for qid in qrel_binary_dict:
        for docid, rel in qrel_binary_dict[qid].items():
            fout.write(qid + ' ' + str(0) + ' ' + str(docid) + ' ' + str(rel) + '\n')


# (dpr) save query & ground truth trec
if not os.path.exists(os.path.join(args.save_dir, 'dpr')):
    os.mkdir(os.path.join(args.save_dir, 'dpr'))

qid2idx = {}
with open(os.path.join(args.save_dir, 'dpr', 'test.query.txt'), 'w') as fout:
    for qid in query_dict:
        if qid not in qid2idx:
            qid2idx[qid] = len(qid2idx)
        fout.write(str(qid2idx[qid]) + '\t' + query_dict[qid] + '\n')

docid2idx = load_pickle(args.docid2idx_file)
with open(os.path.join(args.save_dir, 'dpr', 'test.truth.trec'), 'w') as fout:
    for qid in qrel_dict:
        for docid, rel in qrel_dict[qid].items():
            fout.write(str(qid2idx[qid]) + ' ' + str(0) + ' ' + str(docid2idx[docid]) + ' ' + str(rel) + '\n')

qid2idx2 = {}
with open(os.path.join(args.save_dir, 'dpr', 'test.query.binary.txt'), 'w') as fout:
    for qid in query_binary_dict:
        if qid not in qid2idx2:
            qid2idx2[qid] = len(qid2idx2)
        fout.write(str(qid2idx2[qid]) + '\t' + query_binary_dict[qid] + '\n')

with open(os.path.join(args.save_dir, 'dpr', 'test.truth.binary.trec'), 'w') as fout:
    for qid in qrel_binary_dict:
        for docid, rel in qrel_binary_dict[qid].items():
            fout.write(str(qid2idx2[qid]) + ' ' + str(0) + ' ' + str(docid2idx[docid]) + ' ' + str(rel) + '\n')
