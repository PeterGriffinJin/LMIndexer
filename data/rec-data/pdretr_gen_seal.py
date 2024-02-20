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
from transformers import AutoTokenizer

from IPython import embed

from collections import defaultdict

parser = ArgumentParser()
parser.add_argument('--raw_retrieval_data_dir', type=str, required=True)
parser.add_argument('--raw_item_file', type=str, required=True)
parser.add_argument('--intermediate_dir', type=str, required=True)
parser.add_argument('--bm25_dir', type=str, required=True)
parser.add_argument('--max_rank', type=int, required=True)
parser.add_argument('--domain', type=str, required=True)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--intermediate_item_file', type=str, required=True)

args = parser.parse_args()


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

def text_process(example):

    all_text = []
    # if "title" in example:
    #     all_text.append(f'title: {example["title"]}')
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
meta_dict = {s['asin']:s for s in meta}
for s in meta_dict:
    if "title" not in meta_dict[s]:
        meta_dict[s]["title"] = ""

if not os.path.exists(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval')):
    os.mkdir(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval'))
if not os.path.exists(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval', 'seal')):
    os.mkdir(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval', 'seal'))
print(os.path.join(args.intermediate_dir, args.domain, 'query_retrieval', 'seal'))

save_path = os.path.join(args.intermediate_dir, args.domain, 'query_retrieval', 'seal')


### train set
print(f'-------- Building train file --------')

train_data = defaultdict(dict)
for index, row in tqdm(df_all['train'].iterrows()):
    if 'positive_ctxs' not in train_data[row['query_id']]:
        train_data[row['query_id']]['question'] = row['query']
        train_data[row['query_id']]['positive_ctxs'] = []
        train_data[row['query_id']]['negative_ctxs'] = []
    train_data[row['query_id']]['positive_ctxs'].append(row['product_id'])

with open(os.path.join(args.bm25_dir, 'train.bm25.trec')) as f:
    readin = f.readlines()
    for line in tqdm(readin):
            qid, _, docid, rank, _, _ = line.strip().split()
            if int(qid) not in train_data:
                continue
            try:
                if int(rank) > args.max_rank:
                    continue
                if docid not in set(train_data[int(qid)]['positive_ctxs']):
                    train_data[int(qid)]['negative_ctxs'].append(docid)
            except:
                embed()

for qid in tqdm(train_data):
    train_data[qid]['positive_ctxs'] = [{"text": text_process(meta_dict[docid]),
                                         "title": meta_dict[docid]['title'],
                                         "score": 1000} for docid in train_data[qid]['positive_ctxs']]
    train_data[qid]['negative_ctxs'] = [{"text": text_process(meta_dict[docid]),
                                         "title": meta_dict[docid]['title'],
                                         "score": 0} for docid in train_data[qid]['negative_ctxs']]


### val set
print(f'-------- Building val file --------')

val_data = defaultdict(dict)
for index, row in tqdm(df_all['val'].iterrows()):
    if 'positive_ctxs' not in val_data[row['query_id']]:
        val_data[row['query_id']]['question'] = row['query']
        val_data[row['query_id']]['positive_ctxs'] = []
        val_data[row['query_id']]['negative_ctxs'] = []
    val_data[row['query_id']]['positive_ctxs'].append(row['product_id'])

with open(os.path.join(args.bm25_dir, 'val.bm25.trec')) as f:
    readin = f.readlines()
    for line in tqdm(readin):
            qid, _, docid, rank, _, _ = line.strip().split()
            if int(qid) not in val_data:
                continue
            try:
                if int(rank) > args.max_rank:
                    continue
                if docid not in set(val_data[int(qid)]['positive_ctxs']):
                    val_data[int(qid)]['negative_ctxs'].append(docid)
            except:
                embed()

for qid in tqdm(val_data):
    # val_data[qid]['positive_ctxs'] = [text_process(meta_dict[docid]) for docid in val_data[qid]['positive_ctxs']]
    # val_data[qid]['negative_ctxs'] = [text_process(meta_dict[docid]) for docid in val_data[qid]['negative_ctxs']]
    val_data[qid]['positive_ctxs'] = [{"text": text_process(meta_dict[docid]),
                                         "title": meta_dict[docid]['title'],
                                         "score": 1000} for docid in val_data[qid]['positive_ctxs']]
    val_data[qid]['negative_ctxs'] = [{"text": text_process(meta_dict[docid]),
                                         "title": meta_dict[docid]['title'],
                                         "score": 0} for docid in val_data[qid]['negative_ctxs']]


## save
print(f'-------- Save --------')

with open(os.path.join(save_path, f'{args.domain}-train.json'), 'w') as fout:
    train_data_save = [train_data[qid] for qid in train_data]
    # for qid in tqdm(train_data):
    #     tmp = train_data[qid]
    #     fout.write(json.dumps(tmp)+'\n')
    json.dump(train_data_save, fout, indent = 4)


with open(os.path.join(save_path, f'{args.domain}-dev.json'), 'w') as fout:
    dev_data_save = [val_data[qid] for qid in val_data]
    # for qid in tqdm(val_data):
    #     tmp = val_data[qid]
    #     fout.write(json.dumps(tmp)+'\n')
    json.dump(dev_data_save, fout, indent = 4)


## process corpus

# text processing function
def text_denoise(text):
    p_text = ' '.join(text.split('\r\n'))
    p_text = ' '.join(p_text.split('\n\r'))
    p_text = ' '.join(p_text.split('\n'))
    p_text = ' '.join(p_text.split('\t'))
    p_text = ' '.join(p_text.split('\rm'))
    p_text = ' '.join(p_text.split('\r'))
    p_text = ''.join(p_text.split('$'))
    p_text = ''.join(p_text.split('*'))

    return p_text

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
def preprocess(doc, maxlen=980):
    docid = tokenizer(doc, add_special_tokens=False, truncation=True, max_length=maxlen)['input_ids']
    doc = tokenizer.decode(docid)
    return doc

items = load_json_list(args.intermediate_item_file)
for ii in range(len(items)):
    if 'title' not in items[ii]:
        items[ii]['title'] = ""

items_id2doc = {item['asin']: (preprocess(text_denoise(item['title'])), preprocess(text_denoise(text_process(item)))) for item in items}

# save data
with open(os.path.join(save_path, 'corpus.tsv'), 'w') as fout:
    for asin in rec_datamaps['item2id']:
        fout.write(str(asin) + '\t' + items_id2doc[asin][0] + "\t" + items_id2doc[asin][1] + '\n')

with open(os.path.join(args.bm25_dir, 'test.truth.trec')) as f:
    with open(os.path.join(save_path, 'test.truth.trec'), 'w') as fout:
        readin = f.readlines()
        for line in readin:
            qid, _, docid, _ = line.strip().split()
            fout.write(qid + ' ' + str(0) + ' ' + str(docid) + ' ' + str(1) + '\n')
