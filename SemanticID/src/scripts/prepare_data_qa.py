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

def load_tsv_rows(file_path):
    data = []
    with open(file_path, "r") as f:
        readin = f.readlines()
        for tmp in tqdm(readin):
            data.append(tmp.strip().split('\t'))
    return data

# read input file
print('Reading data...')
# data = []
# with open(args.input_file) as f:
#     readin = f.readlines()
#     for line in tqdm(readin):
#         data.append(json.loads(line))
data = load_tsv_rows(args.input_file)


# write_files
print('Saving data...')
with open(args.output_file, 'w') as fout:
    for d in tqdm(data):
        fout.write(json.dumps({'text': d[1], 'id_length':1}) + '\n')
