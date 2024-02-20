# This code is to merge the TREC_DL_2019 and TREC_DL_2020

import os
import json
import shutil

import random
import pickle
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser

import torch
import random
import numpy as np

from IPython import embed

from collections import defaultdict

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)

args = parser.parse_args()

assert args.mode in ['dpr']

# read queries
with open(os.path.join(args.data_dir, 'TREC_DL_2019', args.mode, 'test.query.txt')) as f:
    query_2019 = f.readlines()
    query_2019_num = len(query_2019)

with open(os.path.join(args.data_dir, 'TREC_DL_2020', args.mode, 'test.query.txt')) as f:
    query_2020 = f.readlines()

if not os.path.exists(os.path.join(args.save_dir, args.mode)):
    os.mkdir(os.path.join(args.save_dir, args.mode))

with open(os.path.join(args.save_dir, args.mode, 'test.query.txt'), 'w') as fout:
    for line in query_2019:
        fout.write(line)
    for line in query_2020:
        tmp = line.strip().split('\t')
        fout.write(str(int(tmp[0]) + query_2019_num) + '\t' + tmp[1] + '\n')

# read qrels
with open(os.path.join(args.data_dir, 'TREC_DL_2019', args.mode, 'test.truth.trec')) as f:
    qrels_2019 = f.readlines()

with open(os.path.join(args.data_dir, 'TREC_DL_2020', args.mode, 'test.truth.trec')) as f:
    qrels_2020 = f.readlines()

with open(os.path.join(args.save_dir, args.mode, 'test.truth.trec'), 'w') as fout:
    for line in qrels_2019:
        fout.write(line)
    for line in qrels_2020:
        tmp = line.strip().split(' ')
        fout.write(str(int(tmp[0]) + query_2019_num) + ' ' + str(0) + ' ' + tmp[2] + ' ' + tmp[3] + '\n')


# read queries binary
with open(os.path.join(args.data_dir, 'TREC_DL_2019', args.mode, 'test.query.binary.txt')) as f:
    query_2019 = f.readlines()
    query_2019_num = len(query_2019)

with open(os.path.join(args.data_dir, 'TREC_DL_2020', args.mode, 'test.query.binary.txt')) as f:
    query_2020 = f.readlines()

with open(os.path.join(args.save_dir, args.mode, 'test.query.binary.txt'), 'w') as fout:
    for line in query_2019:
        fout.write(line)
    for line in query_2020:
        tmp = line.strip().split('\t')
        fout.write(str(int(tmp[0]) + query_2019_num) + '\t' + tmp[1] + '\n')

# read qrels binary
with open(os.path.join(args.data_dir, 'TREC_DL_2019', args.mode, 'test.truth.binary.trec')) as f:
    qrels_2019 = f.readlines()

with open(os.path.join(args.data_dir, 'TREC_DL_2020', args.mode, 'test.truth.binary.trec')) as f:
    qrels_2020 = f.readlines()

with open(os.path.join(args.save_dir, args.mode, 'test.truth.binary.trec'), 'w') as fout:
    for line in qrels_2019:
        fout.write(line)
    for line in qrels_2020:
        tmp = line.strip().split(' ')
        fout.write(str(int(tmp[0]) + query_2019_num) + ' ' + str(0) + ' ' + tmp[2] + ' ' + tmp[3] + '\n')
