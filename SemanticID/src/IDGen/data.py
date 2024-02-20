import glob
import os
import random
from copy import deepcopy
import itertools
from typing import List, Tuple, Dict
from dataclasses import dataclass
from time import time

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling

from .arguments import DataArguments
from .utils import tensorize_batch
from .trainer import DenseIDTrainer

from IPython import embed

@dataclass
class TrainIDCollator(DataCollatorForLanguageModeling):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_seq_length: int = 512
    # encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):

        input_ids_batch = []
        attention_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        all_tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:            
            tgt_len = all_tgt_len - e['id_length']

            e_trunc = self.tokenizer.build_inputs_with_special_tokens(e['token_ids'][:tgt_len])

            self.mlm_probability = self.decoder_mlm_probability
            mask_set = []
            for _ in range(min(len(e_trunc), 256)):
                tmp_mask = self._mask_tokenid_list(e_trunc)  # 1 means masked, 0 means unmasked
                mask_set.append(tmp_mask)

            text_matrix_attention_mask = []
            for i in range(len(e_trunc)):
                idx = random.randint(0, min(len(e_trunc), 256) - 1)
                text_decoder_mlm_mask = deepcopy(mask_set[idx])
                text_decoder_mlm_mask[i] = 1
                text_decoder_mlm_mask = [0] * e['id_length'] + text_decoder_mlm_mask
                text_matrix_attention_mask.append(text_decoder_mlm_mask)
            text_matrix_attention_mask = [text_matrix_attention_mask[0]] + text_matrix_attention_mask

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[-1] = -100
            decoder_labels_batch.append(torch.tensor([-100] + e_trunc))

            decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)

        encoder_batch = {
            "encoder_input_ids": input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
        }
        decoder_batch = {
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
        }

        return encoder_batch, decoder_batch

    def _mask_tokenid_list(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for reconstruction decoder.
        """

        label = deepcopy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full((1, len(label)), self.mlm_probability)
        
        if special_tokens_mask is None:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(label, already_has_special_tokens=True)
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=1.0)
        masked_indices = torch.bernoulli(probability_matrix).squeeze(0).long().tolist()

        return masked_indices


@dataclass
class EvalIDCollator(DataCollatorForLanguageModeling):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_seq_length: int = 512
    # encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):

        assert self.decoder_mlm_probability == 0

        input_ids_batch = []
        attention_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        all_tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:            
            tgt_len = all_tgt_len - e['id_length']

            e_trunc = self.tokenizer.build_inputs_with_special_tokens(e['token_ids'][:tgt_len])

            self.mlm_probability = self.decoder_mlm_probability
            mask_set = []
            # for _ in range(min(len(e_trunc), 256)):
            #     tmp_mask = self._mask_tokenid_list(e_trunc)  # 1 means masked, 0 means unmasked
            #     mask_set.append(tmp_mask)
            tmp_mask = self._mask_tokenid_list(e_trunc)  # 1 means masked, 0 means unmasked
            mask_set.append(tmp_mask)

            text_matrix_attention_mask = []
            for i in range(len(e_trunc)):
                # idx = random.randint(0, min(len(e_trunc), 256) - 1)
                idx = 0
                # text_decoder_mlm_mask = deepcopy(mask_set[idx])
                # text_decoder_mlm_mask[i] = 1
                # text_decoder_mlm_mask = [0] * e['id_length'] + text_decoder_mlm_mask
                text_decoder_mlm_mask = [0] * e['id_length'] + mask_set[idx]     ##################### be careful here, the text decoder mlm is the same for all the tokens (no mask), it is written like this to improve efficiency for embedding/code inference (since decoder is no utilized in inference)
                text_matrix_attention_mask.append(text_decoder_mlm_mask)
            text_matrix_attention_mask = [text_matrix_attention_mask[0]] + text_matrix_attention_mask

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[-1] = -100
            decoder_labels_batch.append(torch.tensor([-100] + e_trunc))

            decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)

        encoder_batch = {
            "encoder_input_ids": input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
        }
        decoder_batch = {
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
        }

        return encoder_batch, decoder_batch

    def _mask_tokenid_list(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for reconstruction decoder.
        """

        label = deepcopy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full((1, len(label)), self.mlm_probability)
        
        if special_tokens_mask is None:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(label, already_has_special_tokens=True)
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=1.0)
        masked_indices = torch.bernoulli(probability_matrix).squeeze(0).long().tolist()

        return masked_indices


class TrainIDDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(TrainIDDataset, self).__init__()
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len

    def __len__(self):  # __len__ is required by huggingface trainer
        count = len(self.dataset)
        return count

    def __getitem__(self, index):
        return self.dataset[index]


class EvalIDDataset(TrainIDDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(EvalIDDataset, self).__init__(tokenizer, data_args, None, cache_dir=cache_dir)

        self.data_files = [data_args.eval_path]
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len

    def __getitem__(self, index):
        return self.dataset[index]


######################### hard neg version #########################

@dataclass
class TrainIDHnCollator(DataCollatorForLanguageModeling):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_seq_length: int = 512
    # encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):

        # query documents
        input_ids_batch = []
        attention_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        # hard negative documents
        key_input_ids_batch = []
        key_attention_mask_batch = []
        key_decoder_labels_batch = []
        key_decoder_matrix_attention_mask_batch = []

        # semantic id token
        semantic_id_batch = []
        key_semantic_id_batch = []

        all_tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:

            # process the query document
            e_trunc, text_matrix_attention_mask = self._process_token_ids(e['token_ids'], all_tgt_len, e['prefix_id'])

            input_ids_batch.append(torch.tensor(e_trunc)) # original tokens, the input to encoder, concatenated with semantic id embeddings to decoder
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc))) # encoder attention mask
            key_input_ids_batch.append(torch.tensor(e_trunc)) # original tokens, the input to encoder, concatenated with semantic id embeddings to decoder
            key_attention_mask_batch.append(torch.tensor([1] * len(e_trunc))) # encoder attention mask
            e_trunc[-1] = -100   # ignore decoding for the special token (which is the last token)
            decoder_labels_batch.append(torch.tensor([-100] * (len(e['prefix_id']) + 1) + e_trunc)) # add ignore for semantic id embeddings
            decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask)) # 1 means unmasked, 0 means masked
            key_decoder_labels_batch.append(torch.tensor([-100] * (len(e['prefix_id']) + 1) + e_trunc)) # add ignore for semantic id embeddings
            key_decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask)) # 1 means unmasked, 0 means masked

            semantic_id_batch.append(e['prefix_id'])
            key_semantic_id_batch.append(e['prefix_id'])

            # process the hard negative documents
            for tk_ids in e['hard_neg_token_ids_list']:
                e_trunc, text_matrix_attention_mask = self._process_token_ids(tk_ids, all_tgt_len, e['prefix_id'])

                key_input_ids_batch.append(torch.tensor(e_trunc)) # original tokens, the input to encoder, concatenated with semantic id embeddings to decoder
                key_attention_mask_batch.append(torch.tensor([1] * len(e_trunc))) # encoder attention mask
                e_trunc[-1] = -100   # ignore decoding for the special token (which is the last token)
                key_decoder_labels_batch.append(torch.tensor([-100] * (len(e['prefix_id']) + 1) + e_trunc)) # add ignore for semantic id embeddings
                key_decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask)) # 1 means unmasked, 0 means masked

                key_semantic_id_batch.append(e['prefix_id'])

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)

        key_input_ids_batch = tensorize_batch(key_input_ids_batch, self.tokenizer.pad_token_id)
        key_attention_mask_batch = tensorize_batch(key_attention_mask_batch, 0)
        key_origin_input_ids_batch = key_input_ids_batch.clone()
        key_decoder_labels_batch = tensorize_batch(key_decoder_labels_batch, -100)
        key_matrix_attention_mask_batch = tensorize_batch(key_decoder_matrix_attention_mask_batch, 0)

        semantic_id_batch = torch.LongTensor(semantic_id_batch)
        key_semantic_id_batch = torch.LongTensor(key_semantic_id_batch)

        # use contrastive or not (to deal with the case when there is no hard negative samples)
        use_contrastive = [e['use_contrastive'] for e in examples]
        use_contrastive_batch = torch.LongTensor(use_contrastive)

        encoder_batch = {
            "encoder_input_ids": input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "key_encoder_input_ids": key_input_ids_batch,
            "key_encoder_attention_mask": key_attention_mask_batch,
            "prefix_semantic_ids": semantic_id_batch,
            "key_prefix_semantic_ids": key_semantic_id_batch,
            "use_contrastive": use_contrastive_batch
        }
        decoder_batch = {
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
            "key_decoder_input_ids": key_origin_input_ids_batch,
            "key_decoder_attention_mask": key_matrix_attention_mask_batch,
            "key_decoder_labels": key_decoder_labels_batch,
        }

        return encoder_batch, decoder_batch

    def _process_token_ids(self, token_ids, all_tgt_len, prefix_id):

        tgt_len = all_tgt_len - (len(prefix_id) + 1)

        e_trunc = self.tokenizer.build_inputs_with_special_tokens(token_ids[:tgt_len])

        self.mlm_probability = self.decoder_mlm_probability
        mask_set = []
        for _ in range(min(len(e_trunc), 256)):
            tmp_mask = self._mask_tokenid_list(e_trunc)  # 1 means masked, 0 means unmasked
            mask_set.append(tmp_mask)

        text_matrix_attention_mask = []
        for i in range(len(e_trunc)):
            idx = random.randint(0, min(len(e_trunc), 256) - 1)
            text_decoder_mlm_mask = deepcopy(mask_set[idx])
            text_decoder_mlm_mask[i] = 1
            text_decoder_mlm_mask = [0] * (len(prefix_id) + 1) + text_decoder_mlm_mask
            text_matrix_attention_mask.append(text_decoder_mlm_mask)
        text_matrix_attention_mask = [text_matrix_attention_mask[0] for _ in range(len(prefix_id) + 1)] + text_matrix_attention_mask  # prepare the decoder mask, add those for semantic id embeddings

        return e_trunc, text_matrix_attention_mask

    def _mask_tokenid_list(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for reconstruction decoder.
        """

        label = deepcopy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full((1, len(label)), self.mlm_probability)
        
        if special_tokens_mask is None:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(label, already_has_special_tokens=True)
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=1.0)
        masked_indices = torch.bernoulli(probability_matrix).squeeze(0).long().tolist()

        return masked_indices


class TrainIDHnDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, trainer: DenseIDTrainer = None, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(TrainIDHnDataset, self).__init__()
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.neg_num = data_args.hn_num
        self.max_len = data_args.max_len
        self.trainer = trainer

    def __len__(self):  # __len__ is required by huggingface trainer
        count = len(self.dataset)
        return count

    def process_fn(self, example, epoch, hashed_seed):
        
        # positive
        encoded_query = example["token_ids"]
        encoded_keys = []
        group_negatives = example["hard_neg_token_ids_list"]

        # assert len(group_negatives) > 0
        if len(group_negatives) == 0:
            use_contrastive = 0
            for _ in range(self.neg_num):
                encoded_keys.append(encoded_query)
        else:
            use_contrastive = 1
            # sample negative
            negative_size = self.neg_num
            if len(group_negatives) < negative_size:
                if hashed_seed is not None:
                    negs = random.choices(group_negatives, k=negative_size)
                else:
                    negs = [x for x in group_negatives]
                    negs = negs * 10
                    negs = negs[:negative_size]
            elif negative_size == 0:
                negs = []
            elif self.data_args.negative_passage_no_shuffle:
                negs = group_negatives[:negative_size]
            else:
                _offset = epoch * negative_size % len(group_negatives)
                negs = [x for x in group_negatives]
                if hashed_seed is not None:
                    random.Random(hashed_seed).shuffle(negs)
                negs = negs * 10
                negs = negs[_offset: _offset + negative_size]

            for neg_psg in negs:
                encoded_keys.append(neg_psg)

        assert len(encoded_keys) == self.neg_num

        # tokenize the prefix ids
        tokenized_prefix_ids = self.tokenizer.encode(example['prefix_ids'], add_special_tokens=False)

        return {"token_ids": encoded_query, "hard_neg_token_ids_list": encoded_keys, "prefix_id": tokenized_prefix_ids, "use_contrastive": use_contrastive}

    def __getitem__(self, index):
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(self.trainer.args.seed)
        return self.process_fn(self.dataset[index], epoch, _hashed_seed)


class EvalIDHnDataset(TrainIDHnDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(EvalIDHnDataset, self).__init__(tokenizer, data_args, None, cache_dir=cache_dir)

        self.data_files = [data_args.eval_path]
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=False, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else self.dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_len = data_args.max_len

    def __getitem__(self, index):
        # return self.dataset[index]
        return self.process_fn(self.dataset[index], 0, None)
