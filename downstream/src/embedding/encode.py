import os
import json
from typing import List, Tuple, Dict
from argparse import ArgumentParser
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, DefaultDataCollator, DataCollatorWithPadding
from datasets import load_dataset

import pickle
from tqdm import tqdm

from IPython import embed

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--tokenizer', type=str, required=False, default='bert-base-uncased')
parser.add_argument('--model_name', type=str, required=False, default='bert-base-uncased')
parser.add_argument('--batch_size', type=int, required=False, default=1)

parser.add_argument('--dataloader_num_workers', type=int, required=False, default=8)
parser.add_argument('--max_length', type=int, default=256)
args = parser.parse_args()


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    # token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@dataclass
class DataCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_len: int = 256

    def __call__(self, features):

        qq = [f["token_id"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {**q_collated}


class TrainDataset(Dataset):

    def __init__(self, tokenizer, cache_dir: str = None) -> None:
        super(TrainDataset, self).__init__()

        tokenizer_name = args.model_name.split('/')[-1]
        self.data_file = os.path.join(args.data_dir, f'meta_all_{tokenizer_name}_tokenized.jsonl')        
        dataset = load_dataset("json", data_files=self.data_file, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = dataset

        self.tokenizer = tokenizer

    def create_one_example(self, text_encoding: List[int]):

        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=args.max_length,
            padding=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        return item

    def __len__(self):  # __len__ is required by huggingface trainer
        count = len(self.dataset)
        return count

    def process_fn(self, example):

        encoded_query = self.create_one_example(example['token_id'])

        return {"token_id": encoded_query}

    def __getitem__(self, index):
        return self.process_fn(self.dataset[index])


tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
model = AutoModel.from_pretrained(args.model_name).cuda()

data = TrainDataset(tokenizer)
data_collator = DataCollator(tokenizer, max_len=args.max_length)
dataloader = DataLoader(data, batch_size=args.batch_size, collate_fn=data_collator, num_workers=args.dataloader_num_workers)

embedding = None

model.eval()
with torch.no_grad():
    for batch_data in tqdm(dataloader):
        batch_data = {k: v.cuda() for k, v in batch_data.items()}
        hiddens = model(**batch_data)

        if args.model_name in ['sentence-transformers/all-mpnet-base-v2']:
            reps = mean_pooling(hiddens.last_hidden_state, batch_data['attention_mask'])
        elif args.model_name in ['bert-base-uncased']:
            reps = hiddens.last_hidden_state[:, 0]
        else:
            raise ValueError('Wrong model name!!!')

        if embedding == None:
            embedding = reps
        else:
            embedding = torch.cat((embedding, reps), 0)

# save
tokenizer_name = args.model_name.split('/')[-1]
torch.save(embedding, os.path.join(args.output_dir, f'{tokenizer_name}-embed.pt'))
