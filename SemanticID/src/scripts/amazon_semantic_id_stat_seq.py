import os
import json
from tqdm import tqdm
from collections import defaultdict, Counter

from IPython import embed


# params
# codebook_size = 512

# init info
domain="sports"

# read documents
data_dir = f'/home/ec2-user/quic-efs/user/bowenjin/SemanticID/data/{domain}/document.json'

docs = []
with open(data_dir) as f:
    readin = f.readlines()
    for tmp in tqdm(readin):
        docs.append(json.loads(tmp))


# read original metadata file
meta_dir = f'/home/ec2-user/quic-efs/user/bowenjin/seq2seq/rec-data/{domain}/preprocess/meta_all.jsonl'

metas = []
with open(meta_dir) as f:
    readin = f.readlines()
    for tmp in tqdm(readin):
        metas.append(json.loads(tmp))


# read code
code_dir = f'/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/{domain}/position2/GB/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/document.3.t5-base.tokenized.json'

semantic_ids = []
with open(code_dir) as f:
    readin = f.readlines()
    for line in tqdm(readin):
        tmp = json.loads(line)
        semantic_ids.append(tmp['prefix_ids'])
semantic_ids_counter = Counter(semantic_ids)

# doc/meta to dict
doc_dict = defaultdict(list)
for cid, doc in tqdm(zip(semantic_ids, docs)):
    doc_dict[(cid)].append(doc)

meta_dict = defaultdict(list)
for cid, doc in tqdm(zip(semantic_ids, metas)):
    meta_dict[(cid)].append(doc)


# useful functions
def extract_brand(meta_list):
    brand_dict = defaultdict(int)
    for m in meta_list:
        if "brand" in m:
            brand_dict[m["brand"]] += 1
    return brand_dict


def extract_category(meta_list):
    category_dict = defaultdict(int)
    for m in meta_list:
        if "categories" in m and len(m['categories']) >= 1:
            category_dict[", ".join(m["categories"][0])] += 1
    return category_dict


def extract_price(meta_list):
    price_dict = defaultdict(int)
    for m in meta_list:
        if "price" in m:
            price_dict[m["price"]] += 1
    return price_dict

# prefix statistics
prefix_dict = defaultdict(list)
prefix_cnt_dict = defaultdict(set)
semantic_id_parse = [ss.split('><') for ss in semantic_ids]
for ss in semantic_id_parse:
    prefix_dict[ss[0]].append(ss[1])
for k in prefix_dict:
    prefix_cnt_dict[k] = Counter(prefix_dict[k])


embed()
