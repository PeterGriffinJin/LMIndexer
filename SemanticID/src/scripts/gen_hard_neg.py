import os
import json
from tqdm import tqdm
from collections import defaultdict, Counter

from argparse import ArgumentParser

from IPython import embed

parser = ArgumentParser()
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--code_file', type=str, required=True)
parser.add_argument('--save_file', type=str, required=True)
args = parser.parse_args()

# read the original tokenized file
docs = []
with open(args.data_file) as f:
    readin = f.readlines()
    for tmp in tqdm(readin):
        docs.append(json.loads(tmp))

# read the semantic ID file
semantic_ids = []
with open(args.code_file) as f:
    readin = f.readlines()
    for tmp in tqdm(readin):
        semantic_ids.append(tmp.strip())

# data structure construction
sid2id_dict = defaultdict(list)
for idd, sid in enumerate(tqdm(semantic_ids)):
    sid2id_dict[sid].append(idd)

# construct the hard neg file
with open(args.save_file, 'w') as fout:
    for i, doc in enumerate(tqdm(docs)):
        sid = semantic_ids[i]
        hard_neg_ids = [neg_id for neg_id in sid2id_dict[sid] if neg_id != i]
        tmp = {
            'token_ids': doc['token_ids'],
            'hard_neg_token_ids_list': [docs[neg_id]['token_ids'] for neg_id in hard_neg_ids],
            'prefix_ids': ''.join([f"<id_{i}_{idd}>" for i, idd in enumerate(sid.split(','))])
        }
        fout.write(json.dumps(tmp)+'\n')
