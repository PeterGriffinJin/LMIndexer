import json
import os
from argparse import ArgumentParser

from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
from multiprocessing import Pool

parser = ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
args = parser.parse_args()


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


# read input file
print('Reading data...')
data = []
with open(args.input_file) as f:
    readin = f.readlines()
    for line in tqdm(readin):
        data.append(text_process(json.loads(line)))

# write_files
print('Saving data...')
with open(args.output_file, 'w') as fout:
    for d in tqdm(data):
        fout.write(json.dumps({'text': d, 'id_length':1}) + '\n')
