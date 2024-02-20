import os
import json
import math
import heapq

import numpy as np

from typing import Any, Callable, List
from collections import defaultdict

import torch
from transformers.generation.logits_process import LogitsProcessor

from IPython import embed

def list2d(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def merge_dict(dict1, dict2):
    return(dict2.update(dict1))

# def sequential_rec_metrics()

def generate_special_token_list(num_codes, codebook_size):
    token_list = []
    for i in range(num_codes):
        token_list = token_list + [f"<id_{str(i)}_{str(k)}>" for k in range(codebook_size)]
    return token_list


class Prefixer():
    def __init__(self, data_args, tokenizer):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self._read_allowed_dict()

    def _read_allowed_dict(self):

        prefix_allowed_dict = defaultdict(set)

        with open(self.data_args.all_id_txt) as f:
            readin = f.readlines()
            for line in readin:
                semantic_id = line.strip()
                tokenized_semantic_id = self.tokenizer.encode(semantic_id)
                tokenized_semantic_id = [self.tokenizer.pad_token_id] + tokenized_semantic_id
                for idd in range(1, len(tokenized_semantic_id)):
                    prefix_allowed_dict[tuple(tokenized_semantic_id[:idd])].add(tokenized_semantic_id[idd])

        self.prefix_allowed_dict = prefix_allowed_dict

    def __call__(self, batch_id, sent):
        return list(self.prefix_allowed_dict[tuple(sent.tolist())])


def save_as_trec(ui_scores, save_trec_dir, save_trec_file, topk):
    with open(os.path.join(save_trec_dir, save_trec_file), "w") as f:
        for qid in ui_scores:
            ui_score = list(ui_scores[qid].items())
            np.random.shuffle(ui_score)  # break ties
            topk_preds = heapq.nlargest(topk, ui_score, key=lambda x: x[1]) # list of k tuples
            docids = [x[0] for x in topk_preds]  # list of k <item_id>
            scores = [x[1] for x in topk_preds]  # list of k <item_id>
            for i, (doc_id, score) in enumerate(zip(docids, scores)):
                f.write("{} Q0 {} {} {} {}\n".format(qid, doc_id, i + 1, score,
                                                     'genai'))
