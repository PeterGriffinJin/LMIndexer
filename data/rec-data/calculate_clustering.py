import os
import json
import gzip
import pickle
import random
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np

from collections import defaultdict

from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score

from IPython import embed

def load_json_list(file_name):
    data = []
    with open(file_name) as f:
        readin = f.readlines()
        for line in readin:
            data.append(json.loads(line))
    return data

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--base', type=str, required=True)
parser.add_argument('--semantic_id_mode', type=str, required=True)

args = parser.parse_args()

base_name = args.base.split('/')[-1]
assert args.semantic_id_mode in ['tree', 'rqvae', 'ours', 'ours-recon3', 'ours-recon2', 'ours-dim256', 'ours-dim128']

# read semantic id
if args.semantic_id_mode not in ['atomic', 'ours', 'ours-weak', 'ours-recon3', 'ours-recon2', 'ours-dim256', 'ours-dim128']:
    rowid2semanticid = json.load(open(os.path.join(args.data_dir, f"{args.semantic_id_mode}-code-{base_name}.json")))
elif args.semantic_id_mode in ['ours', 'ours-weak', 'ours-recon3', 'ours-recon2', 'ours-dim256', 'ours-dim128']:
    rowid2semanticid = json.load(open(os.path.join(args.data_dir, f"{args.semantic_id_mode}-code.json")))

pred_id = []
for i in range(len(rowid2semanticid)):
    tmp_id = rowid2semanticid[str(i+1)].split(',')
    pred_id.append(int(tmp_id[0]))

# read meta
meta = load_json_list(os.path.join(args.data_dir, 'meta_all.jsonl'))

# category statistics
cnt = 0
product_category2id = defaultdict(int)
product_ground_truth = []
for pd in tqdm(meta):
    if len(pd['categories']) != 1:
        cnt += 1
    tmp_type = ','.join(pd['categories'][0])
    if tmp_type not in product_category2id:
        product_category2id[tmp_type] = len(product_category2id)
    product_ground_truth.append(product_category2id[tmp_type])

print(f'Total number of items with more than 1 ground truth category: {cnt}')

ami = adjusted_mutual_info_score(product_ground_truth, pred_id)
mi = mutual_info_score(product_ground_truth, pred_id)

print(f'{args.semantic_id_mode}| ami: {ami}| mi: {mi}')
