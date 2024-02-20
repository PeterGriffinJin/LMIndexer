import json
import os
from argparse import ArgumentParser

from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
from multiprocessing import Pool

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--tokenizer', type=str, required=False, default='t5-base')
parser.add_argument('--minimum-negatives', type=int, required=False, default=1)
parser.add_argument('--mp_chunk_size', type=int, required=False, default=1)
parser.add_argument('--max_length', type=int, default=1024)
args = parser.parse_args()

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

def process(item):

    group = {}
    query = tokenizer.encode(item['text'], add_special_tokens=False, max_length=args.max_length, truncation=True)
    group['token_ids'] = query
    group['id_length'] = item['id_length']

    return json.dumps(group)

tokenizer_name = args.tokenizer.split('/')[-1]

# multiprocessing mode
with open(os.path.join(args.data_dir, f'document.{tokenizer_name}.tokenized.json'), 'w') as f:
    data = []
    with open(os.path.join(args.data_dir, 'document.json')) as fin:
        readin = fin.readlines()
        for line in tqdm(readin):
            data.append(json.loads(line))
    pbar = tqdm(data)
    with Pool() as p:
        for x in p.imap(process, pbar, chunksize=args.mp_chunk_size):
            if x != 0:
                f.write(x + '\n')
