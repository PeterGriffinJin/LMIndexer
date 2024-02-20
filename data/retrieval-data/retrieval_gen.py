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
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--base', type=str, required=True)
parser.add_argument('--semantic_id_mode', type=str, required=True, help='semantic ID mode')
parser.add_argument('--seed', type=int, default=2023)

args = parser.parse_args()
assert args.semantic_id_mode in ['atomic', 'tree', 'rqvae', 'ours']

base_name = args.base.split('/')[-1]

## set seed
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

## tool function definition
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_pickle(data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def load_json_list(file_name):
    data = []
    with open(file_name) as f:
        readin = f.readlines()
        for line in readin:
            data.append(json.loads(line))
    return data

def save_json_list(json_list, file_name):
    with open(file_name, 'w') as fout:
        for line in json_list:
            fout.write(line)

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


## read ID processed data
meta = load_tsv_rows(os.path.join(args.data_dir, "corpus.tsv"))
meta_dict = {s[0]:s for s in meta}
itemid2rowid = {s[0]:(idd+1) for idd, s in enumerate(meta)}
rowid2itemid = {(idd+1):s[0] for idd, s in enumerate(meta)}

semanticid_maxlen=None
if args.semantic_id_mode not in ['atomic', 'ours']:
    rowid2semanticid = json.load(open(os.path.join(args.data_dir, f"{args.semantic_id_mode}-code-{base_name}.json")))
    rowid2semanticid = add_final_id(rowid2semanticid)
    semanticid_maxlen = max([len(s.split(',')) for s in list(rowid2semanticid.values())])
    print(f'Max semantic id length:{semanticid_maxlen}')
elif args.semantic_id_mode == 'ours':
    rowid2semanticid = json.load(open(os.path.join(args.data_dir, f"{args.semantic_id_mode}-code.json")))
    rowid2semanticid = add_final_id(rowid2semanticid)
    semanticid_maxlen = max([len(s.split(',')) for s in list(rowid2semanticid.values())])
    print(f'Max semantic id length:{semanticid_maxlen}')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
if not os.path.exists(os.path.join(args.save_dir, args.semantic_id_mode)):
    os.mkdir(os.path.join(args.save_dir, args.semantic_id_mode))
if not os.path.exists(os.path.join(args.save_dir, args.semantic_id_mode, base_name)) and args.semantic_id_mode not in ['atomic', 'ours']:
    os.mkdir(os.path.join(args.save_dir, args.semantic_id_mode, base_name))
print(os.path.join(args.save_dir, args.semantic_id_mode))

if args.semantic_id_mode in ['atomic', 'ours']:
    save_path = os.path.join(args.save_dir, args.semantic_id_mode)
else:
    save_path = os.path.join(args.save_dir, args.semantic_id_mode, base_name)


## generating data w.r.t semantic ID type
### train set
print(f'-------- Building train file --------')

train_data = []
with open(os.path.join(args.data_dir, 'train.csv')) as f:
    readin = f.readlines()
    for line in tqdm(readin):
        tmp = line.strip().split('\t')
        query, ids = tmp[0], tmp[1].split(',')
        for idd in ids:
            target_text = ''.join(it2semantic_token(idd, args.semantic_id_mode, semanticid_maxlen))
            train_data.append(json.dumps({'source':query, 'target': target_text})+'\n')

save_json_list(train_data, os.path.join(save_path, 'train.json'))


### val set
print(f'-------- Building val file --------')

val_data = []
with open(os.path.join(args.data_dir, 'dev.csv')) as f:
    readin = f.readlines()
    for line in tqdm(readin):
        tmp = line.strip().split('\t')
        query, ids = tmp[0], tmp[1].split(',')
        for idd in ids:
            target_text = ''.join(it2semantic_token(idd, args.semantic_id_mode, semanticid_maxlen))
            val_data.append(json.dumps({'source':query, 'target': target_text})+'\n')

save_json_list(val_data, os.path.join(save_path, 'val.json'))


### test set
print(f'-------- Building test file --------')

test_data = []
with open(os.path.join(args.data_dir, 'test.csv')) as f:
    readin = f.readlines()
    for line in tqdm(readin):
        tmp = line.strip().split('\t')
        query, ids = tmp[0], tmp[1].split(',')
        for idd in ids:
            target_text = ''.join(it2semantic_token(idd, args.semantic_id_mode, semanticid_maxlen))
            test_data.append(json.dumps({'source':query, 'target': target_text})+'\n')

save_json_list(test_data, os.path.join(save_path, 'test.json'))


######################################################################### save all IDs #########################################################################

with open(os.path.join(save_path, 'ids.txt'), 'w') as fout:
    for rowid in range(len(meta)):
        fout.write(''.join(it2semantic_token(rowid2itemid[(rowid+1)], args.semantic_id_mode, semanticid_maxlen)) + '\n')
