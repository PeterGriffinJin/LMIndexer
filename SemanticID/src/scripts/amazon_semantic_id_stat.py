import os
import json
from tqdm import tqdm
from collections import defaultdict, Counter

from IPython import embed


# params
# codebook_size = 512

# read documents
data_dir = '/home/ec2-user/quic-efs/user/bowenjin/SemanticID/data/beauty/document.json'

docs = []
with open(data_dir) as f:
    readin = f.readlines()
    for tmp in tqdm(readin):
        docs.append(json.loads(tmp))


# read original metadata file
meta_dir = '/home/ec2-user/quic-efs/user/bowenjin/seq2seq/rec-data/beauty/preprocess/meta_all.jsonl'

metas = []
with open(meta_dir) as f:
    readin = f.readlines()
    for tmp in tqdm(readin):
        metas.append(json.loads(tmp))


# read code
code_dir = '/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/beauty/position1/GB/load_soft_encdec_kmeans_code/1e-3/0.5/quantization-512/semanticid.txt'

semantic_ids = []
with open(code_dir) as f:
    readin = f.readlines()
    for tmp in tqdm(readin):
        semantic_ids.append(tmp.strip())


# doc/meta to dict
doc_dict = defaultdict(list)
for cid, doc in tqdm(zip(semantic_ids, docs)):
    doc_dict[int(cid)].append(doc)

meta_dict = defaultdict(list)
for cid, doc in tqdm(zip(semantic_ids, metas)):
    meta_dict[int(cid)].append(doc)


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


embed()
