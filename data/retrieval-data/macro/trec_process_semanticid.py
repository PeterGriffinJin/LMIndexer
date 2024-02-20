## This is for generating data for macro1M from trec 2019 & trec 2020 for semantic id methods

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
parser.add_argument('--semanticid_mode', type=str, required=True)
parser.add_argument('--semanticid_data_dir', type=str, required=True)
parser.add_argument('--base', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)

args = parser.parse_args()

assert args.semanticid_mode in ['tree', 'rqvae', 'ours']
base_name = args.base.split('/')[-1]

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_tsv_rows(file_path):
    data = []
    with open(file_path, "r") as f:
        readin = f.readlines()
        for tmp in tqdm(readin):
            data.append(tmp.strip().split('\t'))
    return data

def FixLengthStr2SpecialToken(input_str):
    semantic_ids = input_str.split(',')
    special_tokens = [f"<id_{i}_{k}>" for i, k in enumerate(semantic_ids)]
    return special_tokens

def DiffLengthTreeStr2SpecialToken(input_str, max_len):
    semantic_ids = input_str.split(',')
    special_tokens = [f"<id_{i}_{k}>" if i != (len(semantic_ids)-1) else f"<id_{max_len-1}_{k}>" for i, k in enumerate(semantic_ids)]
    return special_tokens

def add_final_id(itemid2semanticid):
    semanticid_cnt = defaultdict(int)
    for idd in itemid2semanticid:
        tmp = itemid2semanticid[idd]
        itemid2semanticid[idd] = itemid2semanticid[idd] + ',' + str(semanticid_cnt[itemid2semanticid[idd]])
        semanticid_cnt[tmp] += 1
    return itemid2semanticid

def it2semantic_token(inputs, mode, max_len=None):    
    if mode == 'atomic':
        if isinstance(inputs, list):
            return [f'<id_0_{s}>' for s in inputs]
        else:
            return f'<id_0_{inputs}>'
    elif mode in ['rqvae', 'ours']:
        if isinstance(inputs, list):
            return [FixLengthStr2SpecialToken(rowid2semanticid[str(itemid2rowid[s])]) for s in inputs]
        else:
            return FixLengthStr2SpecialToken(rowid2semanticid[str(itemid2rowid[inputs])])
    elif mode == 'tree':
        if isinstance(inputs, list):
            return [DiffLengthTreeStr2SpecialToken(rowid2semanticid[str(itemid2rowid[s])], max_len) for s in inputs]
        else:
            try:
                return DiffLengthTreeStr2SpecialToken(rowid2semanticid[str(itemid2rowid[inputs])], max_len)
            except:
                embed()
    else:
        raise ValueError('Wrong semantic id mode!')

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


## read ID processed data
meta = load_tsv_rows(os.path.join(args.semanticid_data_dir, "corpus.tsv"))
meta_dict = {s[0]:s for s in meta}
itemid2rowid = {s[0]:(idd+1) for idd, s in enumerate(meta)}
rowid2itemid = {(idd+1):s[0] for idd, s in enumerate(meta)}

semanticid_maxlen=None
if args.semanticid_mode not in ['atomic', 'ours']:
    rowid2semanticid = json.load(open(os.path.join(args.semanticid_data_dir, f"{args.semanticid_mode}-code-{base_name}.json")))
    rowid2semanticid = add_final_id(rowid2semanticid)
    semanticid_maxlen = max([len(s.split(',')) for s in list(rowid2semanticid.values())])
    print(f'Max semantic id length:{semanticid_maxlen}')
elif args.semanticid_mode == 'ours':
    rowid2semanticid = json.load(open(os.path.join(args.semanticid_data_dir, f"{args.semanticid_mode}-code.json")))
    rowid2semanticid = add_final_id(rowid2semanticid)
    semanticid_maxlen = max([len(s.split(',')) for s in list(rowid2semanticid.values())])
    print(f'Max semantic id length:{semanticid_maxlen}')


# save query & ground truth trec
if not os.path.exists(os.path.join(args.save_dir, args.semanticid_mode)):
    os.mkdir(os.path.join(args.save_dir, args.semanticid_mode))

qid2idx = {}
with open(os.path.join(args.save_dir, args.semanticid_mode, 'test.query.json'), 'w') as fout:
    for i, qid in enumerate(query_dict):
        assert qid not in qid2idx
        if qid not in qid2idx:
            qid2idx[qid] = len(qid2idx)
        assert qid2idx[qid] == i
        # fout.write(query_dict[qid] + '\t' + str(qid2idx[qid]) + '\n')
        fout.write(json.dumps({
            "source": query_dict[qid],
            "target": str(qid2idx[qid])
        }) + '\n')

with open(os.path.join(args.save_dir, args.semanticid_mode, 'test.truth.trec'), 'w') as fout:
    for qid in qrel_dict:
        for docid, rel in qrel_dict[qid].items():
            fout.write(str(qid2idx[qid]) + ' ' + str(0) + ' ' + ''.join(it2semantic_token(docid, args.semanticid_mode, semanticid_maxlen)) + ' ' + str(rel) + '\n')

qid2idx2 = {}
with open(os.path.join(args.save_dir, args.semanticid_mode, 'test.query.binary.json'), 'w') as fout:
    for i, qid in enumerate(query_binary_dict):
        assert qid not in qid2idx2
        if qid not in qid2idx2:
            qid2idx2[qid] = len(qid2idx2)
        assert qid2idx2[qid] == i
        # fout.write(query_binary_dict[qid] + '\t' + str(qid2idx2[qid]) + '\n')
        fout.write(json.dumps({
            "source": query_binary_dict[qid],
            "target": str(qid2idx2[qid])
        }) + '\n')

with open(os.path.join(args.save_dir, args.semanticid_mode, 'test.truth.binary.trec'), 'w') as fout:
    for qid in qrel_binary_dict:
        for docid, rel in qrel_binary_dict[qid].items():
            fout.write(str(qid2idx2[qid]) + ' ' + str(0) + ' ' + ''.join(it2semantic_token(docid, args.semanticid_mode, semanticid_maxlen)) + ' ' + str(rel) + '\n')
