import csv
import json
from dataclasses import dataclass
from typing import List, Dict

import datasets
from datasets import load_dataset
from transformers import PreTrainedTokenizer, EvalPrediction
import logging
import unicodedata
import regex
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score


logger = logging.getLogger()

from IPython import embed


def compute_perplexity(codes, codebook_size):

    encodings = np.eye(codebook_size)[codes]

    avg_probs = np.mean(encodings, 0)
    perplexity = np.exp(-np.sum(avg_probs * np.log(avg_probs + 1e-10)))
    return perplexity

def tensorize_batch(sequences: List[torch.Tensor], padding_value, align_right=False) -> torch.Tensor:
    if len(sequences[0].size()) == 1:
        max_len_1 = max([s.size(0) for s in sequences])
        out_dims = (len(sequences), max_len_1)
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length_1 = tensor.size(0)
            if align_right:
                out_tensor[i, -length_1:] = tensor
            else:
                out_tensor[i, :length_1] = tensor
        return out_tensor
    elif len(sequences[0].size()) == 2:
        max_len_1 = max([s.size(0) for s in sequences])
        max_len_2 = max([s.size(1) for s in sequences])
        out_dims = (len(sequences), max_len_1, max_len_2)
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length_1 = tensor.size(0)
            length_2 = tensor.size(1)
            if align_right:
                out_tensor[i, -length_1:, :length_2] = tensor
            else:
                out_tensor[i, :length_1, :length_2] = tensor
        return out_tensor
    else:
        raise

# metrics
def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)

    return np.sum(rr_score) / np.sum(y_true)

def mrr_rerank_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)

    return np.max(rr_score)

def read_corpus(path):
    corpus_dict = {}
    with open(path) as f:
        readin = f.readlines()
        for line in readin:
            tmp = line.strip().split('\t')
            corpus_dict[tmp[0]] = tmp[1]
    return corpus_dict

class calculate_metrics():
    def __init__(self, codebook_size=256):
        self.codebook_size = codebook_size

    def __call__(self, evalpred: EvalPrediction):

        scores, labels, codes = evalpred.predictions[-3], evalpred.predictions[-2], evalpred.predictions[-1]

        predictions = scores
        # predictions = np.argmax(scores, -1)
        valid_pred, valid_label = predictions[labels!=-100], labels[labels!=-100]

        prc = (np.sum((valid_pred == valid_label)) / valid_label.shape[0])
        micro_f1 = f1_score(valid_label, valid_pred, average='micro')
        macro_f1 = f1_score(valid_label, valid_pred, average='macro')
        ppl = compute_perplexity(codes, self.codebook_size)

        return {
            "prc": prc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "perplexity": ppl
        }

class calculate_metrics_seq():
    def __init__(self, codebook_size=256):
        self.codebook_size = codebook_size

    def __call__(self, evalpred: EvalPrediction):

        scores, labels, codes = evalpred.predictions[-3], evalpred.predictions[-2], evalpred.predictions[-1]

        predictions = scores
        # predictions = np.argmax(scores, -1)
        valid_pred, valid_label = predictions[labels!=-100], labels[labels!=-100]

        prc = (np.sum((valid_pred == valid_label)) / valid_label.shape[0])
        micro_f1 = f1_score(valid_label, valid_pred, average='micro')
        macro_f1 = f1_score(valid_label, valid_pred, average='macro')
        ppl = compute_perplexity(codes[:,0], self.codebook_size) # calculate ppl for the query document ids

        # assert codes.shape[1] == 2
        # last_diff = (codes[:,0] != codes[:,1]).sum() / codes.shape[0]
        last_diff = sum([len(set(code)) for code in codes.tolist()]) / codes.shape[0] / codes.shape[1]

        return {
            "prc": prc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "perplexity": ppl,
            "same_prefix_last_diff": last_diff,
        }

def calculate_metrics_nocode(evalpred: EvalPrediction):

    scores, labels = evalpred.predictions[-2], evalpred.predictions[-1]

    predictions = scores
    # predictions = np.argmax(scores, -1)
    valid_pred, valid_label = predictions[labels!=-100], labels[labels!=-100]

    prc = (np.sum((valid_pred == valid_label)) / valid_label.shape[0])
    micro_f1 = f1_score(valid_label, valid_pred, average='micro')
    macro_f1 = f1_score(valid_label, valid_pred, average='macro')

    return {
        "prc": prc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }

def save_code_metrics(evalpred: EvalPrediction):

    scores, labels, codes = evalpred.predictions[-3], evalpred.predictions[-2], evalpred.predictions[-1]

    predictions = scores
    # predictions = np.argmax(scores, -1)
    valid_pred, valid_label = predictions[labels!=-100], labels[labels!=-100]

    prc = (np.sum((valid_pred == valid_label)) / valid_label.shape[0])
    micro_f1 = f1_score(valid_label, valid_pred, average='micro')
    macro_f1 = f1_score(valid_label, valid_pred, average='macro')

    return {
        "prc": prc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "codes": codes
    }

def save_code_metrics_seq(evalpred: EvalPrediction):

    scores, labels, codes = evalpred.predictions[-3], evalpred.predictions[-2], evalpred.predictions[-1]

    predictions = scores
    # predictions = np.argmax(scores, -1)
    valid_pred, valid_label = predictions[labels!=-100], labels[labels!=-100]

    prc = (np.sum((valid_pred == valid_label)) / valid_label.shape[0])
    micro_f1 = f1_score(valid_label, valid_pred, average='micro')
    macro_f1 = f1_score(valid_label, valid_pred, average='macro')

    return {
        "prc": prc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "codes": np.squeeze(codes)
    }

def save_embed_metrics(evalpred: EvalPrediction):

    assert len(evalpred.predictions) == 7 or len(evalpred.predictions) == 8

    return {
        "embed": evalpred.predictions[0]
    }

# def generate_special_token_list(num_codes, codebook_size):
#     token_list = []
#     for i in range(num_codes):
#         token_list = token_list + [f"<id_{str(i)}_{str(k)}>" for k in range(codebook_size)]
#     return token_list

def generate_special_token_list(code_pos, codebook_size):
    token_list = [f"<id_{str(code_pos)}_{str(k)}>" for k in range(codebook_size)]
    return token_list
