import os
import json
from tqdm import tqdm
from collections import defaultdict, Counter

from IPython import embed

# params
# codebook_size = 512

# init info
domain="Trivia"

# read documents
data_dir = f'/home/ubuntu/quic-efs/user/bowenjin/SemanticID/data/{domain}/document.json'

docs = []
with open(data_dir) as f:
    readin = f.readlines()
    for tmp in tqdm(readin):
        docs.append(json.loads(tmp))

# read code
# code_dir = f'/home/ubuntu/quic-efs/user/bowenjin/SemanticID/ckpt/{domain}/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-4/0.7/quantization-512/document.3.t5-base.tokenized.json'
code_dir = f'/home/ubuntu/quic-efs/user/bowenjin/seq2seq/retrieval-data/processed/{domain}/ours/ids.txt'

semantic_ids = []
with open(code_dir) as f:
    readin = f.readlines()
    for line in tqdm(readin):
        # tmp = json.loads(line)
        # semantic_ids.append(tmp['prefix_ids'])
        tmp = line.strip()
        semantic_ids.append(tmp)
semantic_ids_counter = Counter(semantic_ids)

# doc/meta to dict
doc_dict = defaultdict(list)
for cid, doc in tqdm(zip(semantic_ids, docs)):
    doc_dict[(cid)].append(doc)

# prefix statistics
prefix_dict = defaultdict(list)
prefix_cnt_dict = defaultdict(set)
semantic_id_parse = [ss.split('><') for ss in semantic_ids]
for ss in semantic_id_parse:
    prefix_dict[ss[0]].append(ss[1])
for k in prefix_dict:
    prefix_cnt_dict[k] = Counter(prefix_dict[k])


embed()
