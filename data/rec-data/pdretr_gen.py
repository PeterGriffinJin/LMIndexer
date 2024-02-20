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
parser.add_argument('--raw_retrieval_data_dir', type=str, required=True)
parser.add_argument('--raw_item_file', type=str, required=True)
parser.add_argument('--intermediate_dir', type=str, required=True)
parser.add_argument('--domain', type=str, required=True)
parser.add_argument('--base', type=str, required=True)
parser.add_argument('--semantic_id_mode', type=str, required=True, help='semantic ID mode')
parser.add_argument('--seed', type=int, default=2023)

args = parser.parse_args()
assert args.semantic_id_mode in ['atomic', 'tree', 'rqvae', 'ours', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']

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
    elif mode in ['rqvae', 'ours', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']:
        if isinstance(inputs, list):
            return [FixLengthStr2SpecialToken(rowid2semanticid[str(itemid2rowid[s])]) for s in inputs]
        else:
            return FixLengthStr2SpecialToken(rowid2semanticid[str(itemid2rowid[inputs])])
    elif mode == 'tree':
        if isinstance(inputs, list):
            return [DiffLengthTreeStr2SpecialToken(rowid2semanticid[str(itemid2rowid[s])], max_len) for s in inputs]
        else:
            return DiffLengthTreeStr2SpecialToken(rowid2semanticid[str(itemid2rowid[inputs])], max_len)
    else:
        raise ValueError('Wrong semantic id mode!')

def text_process(example):

    all_text = []
    if "title" in example:
        all_text.append(f'title: {example["title"]}')
    if "description" in example:
        all_text.append(f'description: {example["description"]}')
    if "brand" in example:
        all_text.append(f'brand: {example["brand"]}')
    if "price" in example:
        all_text.append(f'price: {example["price"]}')
    if "categories" in example and len(example["categories"][0]) > 0:
        all_text.append(f'category: {", ".join(example["categories"][0])}')
    return ', '.join(all_text)

## read retrieval items
df_examples = pd.read_parquet(os.path.join(args.raw_retrieval_data_dir, 'shopping_queries_dataset_examples.parquet'))
df_products = pd.read_parquet(os.path.join(args.raw_retrieval_data_dir, 'shopping_queries_dataset_products.parquet'))
# df_sources = pd.read_csv(os.path.join(args.raw_retrieval_data_dir, "shopping_queries_dataset_sources.csv"))


## match items
if not os.path.exists(os.path.join(args.intermediate_dir, 'overlap_item.json')):
    metadata = []
    with open(args.raw_item_file) as f:
        readin = f.readlines()
        for tmp in tqdm(readin):
            metadata.append(eval(tmp))

    metadata_dict = {}
    for meta in tqdm(metadata):
        metadata_dict[meta['asin']] = meta

    overlap = []
    for pid in tqdm(df_products['product_id']):
        if pid in metadata_dict:
            overlap.append(metadata_dict[pid])

    print(f'Writing the intermediate data to {os.path.join(args.intermediate_dir, "overlap_item.json")}')
    with open(os.path.join(args.intermediate_dir, 'overlap_item.json'), 'w') as fout:
        json.dump(overlap, fout, indent = 6)
else:
    print(f'Reading the intermediate data from {os.path.join(args.intermediate_dir, "overlap_item.json")}')
    overlap = json.load(open(os.path.join(args.intermediate_dir, 'overlap_item.json')))


## statistics on category (domain)
overlap_categories = defaultdict(int)
overlap_list = []

for o in tqdm(overlap):
    if 'categories' not in o:
        continue
    overlap_categories[o['categories'][0][0]] += 1

for k, v in overlap_categories.items():
    overlap_list.append((k, v))

overlap_list.sort(key=lambda x: -x[1])

print(f'Number of overlapped categories:{len(overlap_categories)}')
print(f'Overlapped category item statistics:{overlap_list}')


## data processing & train/val/test split
try:
    rec_datamaps = load_json(os.path.join(args.intermediate_dir, args.domain, "preprocess/datamaps.json"))
except:
    raise ValueError('You should conduct data processing for recommendation first!')

if not os.path.exists(os.path.join(args.intermediate_dir, args.domain, "preprocess/retrieval.pkl")):

    ## Filter queries
    df_examples_rec = df_examples.loc[df_examples['product_id'].isin(list(rec_datamaps['item2id'].keys()))]
    df_examples_rec = df_examples_rec.loc[df_examples_rec['esci_label'] == 'E']


    ## Split train/val/test
    df_train_val = df_examples_rec.loc[df_examples_rec['split'] == 'train']
    train_val_qids = list(set(df_train_val['query_id'].tolist()))
    random.shuffle(train_val_qids)
    x_qids_train, x_qids_val = train_val_qids[:int(len(train_val_qids) * 7 / 8)], train_val_qids[int(len(train_val_qids) * 7 / 8):]

    df_train = df_train_val.loc[df_train_val['query_id'].isin(x_qids_train)]
    df_val = df_train_val.loc[df_train_val['query_id'].isin(x_qids_val)]
    df_test = df_examples_rec.loc[df_examples_rec['split'] == 'test']
    assert len(df_train) + len(df_val) + len(df_test) == len(df_examples_rec)
    print(f'# Pair labels | train:{len(df_train)}, val:{len(df_val)}, test:{len(df_test)}')
    print(f'# Queries | train:{len(set(df_train["query_id"].tolist()))}, val:{len(set(df_val["query_id"].tolist()))}, test:{len(set(df_test["query_id"].tolist()))}')

    df_all = {
        'train': df_train,
        'val': df_val,
        'test': df_test,
    }

    ## save
    print(f'Saving the {args.domain} domain retrieval processed data to {os.path.join(args.intermediate_dir, args.domain, "preprocess/retrieval.pkl")}')
    save_pickle(df_all, os.path.join(args.intermediate_dir, args.domain, "preprocess/retrieval.pkl"))

else:
    ## load
    print(f'Loading the {args.domain} domain retrieval processed data from {os.path.join(args.intermediate_dir, args.domain, "preprocess/retrieval.pkl")}')
    df_all = load_pickle(os.path.join(args.intermediate_dir, args.domain, "preprocess/retrieval.pkl"))
    print(f'# Pair labels | train:{len(df_all["train"])}, val:{len(df_all["val"])}, test:{len(df_all["test"])}')
    print(f'# Queries | train:{len(set(df_all["train"]["query_id"].tolist()))}, val:{len(set(df_all["val"]["query_id"].tolist()))}, test:{len(set(df_all["test"]["query_id"].tolist()))}')

## read ID processed data
meta = load_json_list(os.path.join(args.intermediate_dir, args.domain, "preprocess/meta_all.jsonl"))
meta_dict = {s['item_id']:s for s in meta}
itemid2rowid = {s['item_id']:(idd+1) for idd, s in enumerate(meta)}
rowid2itemid = {(idd+1):s['item_id'] for idd, s in enumerate(meta)}

semanticid_maxlen=None
if args.semantic_id_mode not in ['atomic', 'ours', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']:
    rowid2semanticid = json.load(open(os.path.join(args.intermediate_dir, args.domain, 'preprocess', f"{args.semantic_id_mode}-code-{base_name}.json")))
    rowid2semanticid = add_final_id(rowid2semanticid)
    semanticid_maxlen = max([len(s.split(',')) for s in list(rowid2semanticid.values())])
    print(f'Max semantic id length:{semanticid_maxlen}')
elif args.semantic_id_mode in ['ours', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']:
    rowid2semanticid = json.load(open(os.path.join(args.intermediate_dir, args.domain, 'preprocess', f"{args.semantic_id_mode}-code.json")))
    rowid2semanticid = add_final_id(rowid2semanticid)
    semanticid_maxlen = max([len(s.split(',')) for s in list(rowid2semanticid.values())])
    print(f'Max semantic id length:{semanticid_maxlen}')

if not os.path.exists(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval')):
    os.mkdir(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval'))
if not os.path.exists(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval', args.semantic_id_mode)):
    os.mkdir(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval', args.semantic_id_mode))
if not os.path.exists(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval', args.semantic_id_mode, base_name)) and args.semantic_id_mode not in ['atomic', 'ours', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']:
    os.mkdir(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval', args.semantic_id_mode, base_name))
print(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval', args.semantic_id_mode))

if args.semantic_id_mode in ['atomic', 'ours', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']:
    save_path = os.path.join(args.intermediate_dir, args.domain, 'query_retrieval', args.semantic_id_mode)
else:
    save_path = os.path.join(args.intermediate_dir, args.domain, 'query_retrieval', args.semantic_id_mode, base_name)


## generating data w.r.t semantic ID type
### train set
print(f'-------- Building train file --------')

train_data = []
for index, row in tqdm(df_all['train'].iterrows()):
    target_text = ''.join(it2semantic_token(rec_datamaps['item2id'][row['product_id']], args.semantic_id_mode, semanticid_maxlen))
    train_data.append(json.dumps({'source':row['query'], 'target': target_text})+'\n')

save_json_list(train_data, os.path.join(save_path, 'train.json'))

### val set
print(f'-------- Building val file --------')

val_data = []
for index, row in tqdm(df_all['val'].iterrows()):
    target_text = ''.join(it2semantic_token(rec_datamaps['item2id'][row['product_id']], args.semantic_id_mode, semanticid_maxlen))
    val_data.append(json.dumps({'source':row['query'], 'target': target_text})+'\n')

save_json_list(val_data, os.path.join(save_path, 'val.json'))


### test set
print(f'-------- Building test file --------')

test_data = []
for index, row in tqdm(df_all['test'].iterrows()):
    target_text = ''.join(it2semantic_token(rec_datamaps['item2id'][row['product_id']], args.semantic_id_mode, semanticid_maxlen))
    test_data.append(json.dumps({'source':row['query'], 'target': target_text})+'\n')

save_json_list(test_data, os.path.join(save_path, 'test.json'))


### copy the valid ID file
if args.semantic_id_mode in ['atomic', 'ours', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']:
    ID_rec_source = os.path.join(args.intermediate_dir, args.domain, 'sequential_retrieval', args.semantic_id_mode, 'ids.txt')
else:
    ID_rec_source = os.path.join(args.intermediate_dir, args.domain, 'sequential_retrieval', args.semantic_id_mode, base_name, 'ids.txt')

shutil.copyfile(ID_rec_source, os.path.join(save_path, 'ids.txt'))
