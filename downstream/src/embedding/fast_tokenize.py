import json
import os
from argparse import ArgumentParser
import pickle

from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
from multiprocessing import Pool

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

def load_json_rows(file_path):
    data = []
    with open(file_path, "r") as f:
        readin = f.readlines()
        for tmp in tqdm(readin):
            data.append(eval(tmp))
    return data

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


def process(item):

    # group = {k:v for k, v in item.items()}
    group = {}
    query = tokenizer.encode(text_process(item), add_special_tokens=False, max_length=args.max_length, truncation=True)
    group['token_id'] = query
    group['text'] = text_process(item)
    group['item_id'] = item['item_id']

    return json.dumps(group)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_json(list_of_dict, file):
    with open(file, 'w') as fout:
        for tmp in list_of_dict:
            fout.write(json.dumps(tmp)+'\n')

# read data
data = load_json_rows(os.path.join(args.input_dir, 'meta_all.jsonl'))

# multiprocessing mode
tokenizer_name = args.tokenizer.split('/')[-1]
with open(os.path.join(args.output, f'meta_all_{tokenizer_name}_tokenized.jsonl'), 'w') as f:
    pbar = tqdm(data)
    with Pool() as p:
        for x in p.imap(process, pbar, chunksize=args.mp_chunk_size):
            if x != 0:
                f.write(x + '\n')
