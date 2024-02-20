import os
import json
import gzip
import pickle
import random
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np

from collections import defaultdict

from IPython import embed

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--base', type=str, required=True)
parser.add_argument('--semantic_id_mode', type=str, required=True)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--train_iter', type=int, default=5)

args = parser.parse_args()

assert args.semantic_id_mode in ['atomic', 'tree', 'rqvae', 'ours', 'ours-weak', 'rqvae-weak-1', 'rqvae-weak-2', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def read_json(file):
    data = []
    with open(file) as f:
        readin = f.readlines()
    for line in tqdm(readin):
        data.append(json.loads(line))
    return data

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def random_pick_template(templates):
    template_list = [templates[t] for t in templates]
    return template_list[random.randint(0, len(template_list)-1)]

def save_json_list(json_list, file_name):
    with open(file_name, 'w') as fout:
        for line in json_list:
            fout.write(line)

def load_json_list(file_name):
    data = []
    with open(file_name) as f:
        readin = f.readlines()
        for line in readin:
            data.append(json.loads(line))
    return data

def add_final_id(itemid2semanticid):
    semanticid_cnt = defaultdict(int)
    for idd in itemid2semanticid:
        tmp = itemid2semanticid[idd]
        itemid2semanticid[idd] = itemid2semanticid[idd] + ',' + str(semanticid_cnt[itemid2semanticid[idd]])
        semanticid_cnt[tmp] += 1
    return itemid2semanticid, semanticid_cnt

def FixLengthStr2SpecialToken(input_str):
    semantic_ids = input_str.split(',')
    special_tokens = [f"<id_{i}_{k}>" for i, k in enumerate(semantic_ids)]
    return special_tokens

def DiffLengthTreeStr2SpecialToken(input_str, max_len):
    semantic_ids = input_str.split(',')
    special_tokens = [f"<id_{i}_{k}>" if i != (len(semantic_ids)-1) else f"<id_{max_len-1}_{k}>" for i, k in enumerate(semantic_ids)]
    return special_tokens

def it2semantic_token(inputs, mode, max_len=None):    
    if mode == 'atomic':
        if isinstance(inputs, list):
            return [f'<id_0_{s}>' for s in inputs]
        else:
            return f'<id_0_{inputs}>'
    elif mode in ['rqvae', 'ours', 'ours-weak', 'rqvae-weak-1', 'rqvae-weak-2', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']:
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

base_name = args.base.split('/')[-1]

# real preprocess files
sequential_data = ReadLineFromFile(os.path.join(args.data_dir, 'preprocess', 'sequential_data.txt'))
negative_samples = ReadLineFromFile(os.path.join(args.data_dir, 'preprocess', 'negative_samples.txt'))
datamaps = load_json(os.path.join(args.data_dir, 'preprocess', 'datamaps.json'))
meta = load_json_list(os.path.join(args.data_dir, 'preprocess', 'meta_all.jsonl'))
meta_dict = {s['item_id']:s for s in meta}
itemid2rowid = {s['item_id']:(idd+1) for idd, s in enumerate(meta)}
rowid2itemid = {(idd+1):s['item_id'] for idd, s in enumerate(meta)}

# read semantic id
semanticid_maxlen=None
if args.semantic_id_mode not in ['atomic', 'ours', 'ours-weak', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']:
    rowid2semanticid = json.load(open(os.path.join(args.data_dir, 'preprocess', f"{args.semantic_id_mode}-code-{base_name}.json")))
    rowid2semanticid, semanticid_cnt = add_final_id(rowid2semanticid)
    semanticid_maxlen = max([len(s.split(',')) for s in list(rowid2semanticid.values())])
    print(f'Max semantic id length:{semanticid_maxlen}')
    print(f'Max last position id:{max(list(semanticid_cnt.values()))}')
elif args.semantic_id_mode in ['ours', 'ours-weak', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']:
    rowid2semanticid = json.load(open(os.path.join(args.data_dir, 'preprocess', f"{args.semantic_id_mode}-code.json")))
    rowid2semanticid, semanticid_cnt = add_final_id(rowid2semanticid)
    semanticid_maxlen = max([len(s.split(',')) for s in list(rowid2semanticid.values())])
    print(f'Max semantic id length:{semanticid_maxlen}')
    print(f'Max last position id:{max(list(semanticid_cnt.values()))}')

if not os.path.exists(os.path.join(args.data_dir, 'sequential_retrieval')):
    os.mkdir(os.path.join(args.data_dir, 'sequential_retrieval'))
if not os.path.exists(os.path.join(args.data_dir, 'sequential_retrieval', args.semantic_id_mode)):
    os.mkdir(os.path.join(args.data_dir, 'sequential_retrieval', args.semantic_id_mode))
if not os.path.exists(os.path.join(args.data_dir, 'sequential_retrieval', args.semantic_id_mode, base_name)) and args.semantic_id_mode not in ['atomic', 'ours', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']:
    os.mkdir(os.path.join(args.data_dir, 'sequential_retrieval', args.semantic_id_mode, base_name))
print(os.path.join(args.data_dir, 'sequential_retrieval', args.semantic_id_mode))

if args.semantic_id_mode in ['atomic', 'ours', 'ours-weak', 'ours-recon2', 'ours-recon3', 'ours-dim256', 'ours-dim128']:
    save_path = os.path.join(args.data_dir, 'sequential_retrieval', args.semantic_id_mode)
else:
    save_path = os.path.join(args.data_dir, 'sequential_retrieval', args.semantic_id_mode, base_name)


######################################################################### pre-sampling #########################################################################
### train
if not os.path.exists(os.path.join(args.data_dir, 'sequential_retrieval', f'train_sampling_iter_{args.train_iter}.txt')):
    with open(os.path.join(args.data_dir, 'sequential_retrieval', f'train_sampling_iter_{args.train_iter}.txt'), 'w') as fout:
        for _ in range(args.train_iter):
            for sequential_datum in sequential_data:
                sequence = sequential_datum.split()
                # user_id = sequence[0]

                end_candidates = [_ for _ in range(max(2, len(sequence) - 6), len(sequence) - 3)]
                end_index = random.randint(0, len(end_candidates)-1)
                end_pos = end_candidates[end_index]
                start_candidates = [_ for _ in range(1, min(4, end_pos))]
                start_index = random.randint(0, len(start_candidates)-1)
                start_pos = start_candidates[start_index]
                purchase_history = sequence[start_pos:end_pos+1] # sample a history sequence from the full user purchase history
                target_item = sequence[end_pos+1]
                fout.write(','.join(purchase_history) + '\t' + target_item + '\n')
    print(f"Finish sampling train data and saved to {os.path.join(args.data_dir, 'sequential_retrieval', f'train_sampling_iter_{args.train_iter}.txt')}")
else:
    print(f"Reading presampled train data from {os.path.join(args.data_dir, 'sequential_retrieval', f'train_sampling_iter_{args.train_iter}.txt')}")


### val
if not os.path.exists(os.path.join(args.data_dir, 'sequential_retrieval', 'val_sampling.txt')):
    with open(os.path.join(args.data_dir, 'sequential_retrieval', 'val_sampling.txt'), 'w') as fout:
        for sequential_datum in tqdm(sequential_data):
            sequence = sequential_datum.split()
            purchase_history = sequence[1:-2]
            target_item = sequence[-2]
            fout.write(','.join(purchase_history) + '\t' + target_item + '\n')
    print(f"Finish sampling val data and saved to {os.path.join(args.data_dir, 'sequential_retrieval', 'val_sampling.txt')}")
else:
    print(f"Reading presampled val data from {os.path.join(args.data_dir, 'sequential_retrieval', 'val_sampling.txt')}")


### test
if not os.path.exists(os.path.join(args.data_dir, 'sequential_retrieval', 'test_sampling.txt')):
    with open(os.path.join(args.data_dir, 'sequential_retrieval', 'test_sampling.txt'), 'w') as fout:
        for sequential_datum in tqdm(sequential_data):
            sequence = sequential_datum.split()
            purchase_history = sequence[1:-1]
            target_item = sequence[-1]
            fout.write(','.join(purchase_history) + '\t' + target_item + '\n')
    print(f"Finish sampling test data and saved to {os.path.join(args.data_dir, 'sequential_retrieval', 'test_sampling.txt')}")
else:
    print(f"Reading presampled test data from {os.path.join(args.data_dir, 'sequential_retrieval', 'test_sampling.txt')}")


######################################################################### save all IDs #########################################################################

with open(os.path.join(save_path, 'ids.txt'), 'w') as fout:
    for rowid in range(len(meta)):
        fout.write(''.join(it2semantic_token(rowid2itemid[(rowid+1)], args.semantic_id_mode, semanticid_maxlen)) + '\n')

######################################################################### generating final files #########################################################################
### train set
print(f'-------- Building train file (Iter {args.train_iter}) --------')

train_data = []
with open(os.path.join(args.data_dir, 'sequential_retrieval', f'train_sampling_iter_{args.train_iter}.txt')) as f:
    readin = f.readlines()
    for line in tqdm(readin):
        purchase_history, target_item = line.strip().split('\t')
        purchase_history = purchase_history.split(',')

        source_text = [text_process(meta_dict[idd]) for idd in purchase_history[::-1]] ### make last purchased item appear at first, to avoid truncation
        source_text = '. '.join(source_text)
        target_text = ''.join(it2semantic_token(target_item, args.semantic_id_mode, semanticid_maxlen))

        train_data.append(json.dumps({'source':source_text, 'target': target_text})+'\n')

save_json_list(train_data, os.path.join(save_path, 'train.json'))

### val set
print(f'-------- Building val file --------')

val_data = []
with open(os.path.join(args.data_dir, 'sequential_retrieval', 'val_sampling.txt')) as f:
    readin = f.readlines()
    for line in tqdm(readin):
        purchase_history, target_item = line.strip().split('\t')
        purchase_history = purchase_history.split(',')

        source_text = [text_process(meta_dict[idd]) for idd in purchase_history[::-1]] ### make last purchased item appear at first, to avoid truncation
        source_text = '. '.join(source_text)
        target_text = ''.join(it2semantic_token(target_item, args.semantic_id_mode, semanticid_maxlen))

        val_data.append(json.dumps({'source':source_text, 'target': target_text})+'\n')

save_json_list(val_data, os.path.join(save_path, 'val.json'))


### test set
print(f'-------- Building test file --------')

test_data = []
with open(os.path.join(args.data_dir, 'sequential_retrieval', 'test_sampling.txt')) as f:
    readin = f.readlines()
    for line in tqdm(readin):
        purchase_history, target_item = line.strip().split('\t')
        purchase_history = purchase_history.split(',')

        source_text = [text_process(meta_dict[idd]) for idd in purchase_history[::-1]] ### make last purchased item appear at first, to avoid truncation
        source_text = '. '.join(source_text)
        target_text = ''.join(it2semantic_token(target_item, args.semantic_id_mode, semanticid_maxlen))

        test_data.append(json.dumps({'source':source_text, 'target': target_text})+'\n')

save_json_list(test_data, os.path.join(save_path, 'test.json'))
