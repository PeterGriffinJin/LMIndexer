import json
import random
import argparse
import os
import datasets
from collections import defaultdict

from tqdm import tqdm

from IPython import embed

random.seed(313)

def read_query(file):
    dict = {}
    with open(file, 'r') as f:
        for line in f:
            qid, text = line.split('\t')
            dict[qid] = text.strip()
    return dict

def read_qrel_dev(file):
    dict = {}
    with open(file, 'r') as f:
        for line in f:
            qid, docid = line.split('\t')
            docid = int(docid)
            if docid not in dict:
                dict[docid] = [qid]
            else:
                dict[docid].append(qid)
    return dict

def read_qrel(file):
    dict = {}
    with open(file, 'r') as f:
        for line in f:
            qid, _, docid, _ = line.split('\t')
            docid = int(docid)
            if docid not in dict:
                dict[docid] = [qid]
            else:
                dict[docid].append(qid)
    return dict

parser = argparse.ArgumentParser()
parser.add_argument("--train_num", type=int)
parser.add_argument("--eval_num", type=int, default=6980)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()

print("Creating MS MARCO dataset...")

NUM_TRAIN = args.train_num
NUM_EVAL = args.eval_num
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

data = datasets.load_dataset('Tevatron/msmarco-passage-corpus', cache_dir='cache')['train']
corpus = [item for item in data]
random.shuffle(corpus)
dev_query = read_query('/home/ubuntu/quic-efs/user/bowenjin/seq2seq/retrieval-data/marco/dev.query.txt')
dev_qrel = read_qrel_dev('/home/ubuntu/quic-efs/user/bowenjin/seq2seq/retrieval-data/marco/qrels.dev.tsv')
train_query = read_query('/home/ubuntu/quic-efs/user/bowenjin/seq2seq/retrieval-data/marco/train.query.txt')
train_qrel = read_qrel('/home/ubuntu/quic-efs/user/bowenjin/seq2seq/retrieval-data/marco/qrels.train.tsv')
print('Finish reading files!')

train_ids = list(train_qrel.keys())
random.shuffle(train_ids)
dev_ids = list(set(dev_qrel.keys()).difference(set(train_qrel.keys())))  # make sure no data leakage
random.shuffle(dev_ids)

assert NUM_TRAIN > len(train_ids)

exist_docid_set = set()
train_q2docid = defaultdict(set)
test_q2docid = defaultdict(set)
corpus_data = []

print("Processing training data")
current_ind = 0
for docid in tqdm(train_ids):
    passage = data[docid]['text']

    for qid in train_qrel[docid]:
        question = train_query[qid]
        train_q2docid[question].add(docid)

    corpus_data.append(f"{data[docid]['docid']}\t{passage}")
    exist_docid_set.add(docid)
    current_ind += 1

print("Processing testing data")
for docid in tqdm(dev_ids):
    passage = data[docid]['text']

    for qid in dev_qrel[docid]:
        question = dev_query[qid]
        test_q2docid[question].add(docid)

    if docid not in exist_docid_set:
        corpus_data.append(f"{data[docid]['docid']}\t{passage}")
        exist_docid_set.add(docid)
        current_ind += 1

for item in tqdm(corpus):
    if current_ind >= NUM_TRAIN:
        break
    if int(item['docid']) in exist_docid_set:
        continue
    passage = item['text']

    corpus_data.append(f"{item['docid']}\t{passage}")
    current_ind += 1

with open(f'{args.save_dir}/train.csv', 'w') as tf:
    for question in train_q2docid:
        tf.write(question + '\t' + ','.join([str(iid) for iid in list(train_q2docid[question])]) + '\n')

with open(f'{args.save_dir}/test.csv', 'w') as df:
    for question in test_q2docid:
        df.write(question + '\t' + ','.join([str(iid) for iid in list(test_q2docid[question])]) + '\n')

with open(f'{args.save_dir}/corpus.tsv', 'w') as fout:
    for item in corpus_data:
        fout.write(item + '\n')
