# This is the code script to tokenize the NQ and TriviaQA datasets.

import json
import os
from argparse import ArgumentParser
import pickle

from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
from multiprocessing import Pool

from IPython import embed

parser = ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--tokenizer', type=str, required=False, default='bert-base-uncased')
parser.add_argument('--minimum-negatives', type=int, required=False, default=1)
parser.add_argument('--mp_chunk_size', type=int, required=False, default=1)
parser.add_argument('--max_length', type=int, default=256)
args = parser.parse_args()

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

save_dir = os.path.split(args.output)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

files = os.listdir(args.input_dir)

def load_tsv_rows(file_path):
    data = []
    with open(file_path, "r") as f:
        readin = f.readlines()
        for tmp in tqdm(readin):
            data.append(tmp.strip().split('\t'))
    return data

def process(item):

    assert len(item) == 2

    group = {}
    query = tokenizer.encode(item[1], add_special_tokens=False, max_length=args.max_length, truncation=True)
    group['token_id'] = query
    group['text'] = item[1]
    group['item_id'] = item[0]

    return json.dumps(group)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_json(list_of_dict, file):
    with open(file, 'w') as fout:
        for tmp in list_of_dict:
            fout.write(json.dumps(tmp)+'\n')

# read data
data = load_tsv_rows(os.path.join(args.input_dir, 'corpus.tsv'))

# multiprocessing mode
tokenizer_name = args.tokenizer.split('/')[-1]
with open(os.path.join(args.output, f'meta_all_{tokenizer_name}_tokenized.jsonl'), 'w') as f:
    pbar = tqdm(data)
    with Pool() as p:
        for x in p.imap(process, pbar, chunksize=args.mp_chunk_size):
            if x != 0:
                f.write(x + '\n')
