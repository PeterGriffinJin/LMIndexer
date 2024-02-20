import json
import os
import logging
import copy
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

from transformers import AutoModel, BatchEncoding, PreTrainedModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import ModelOutput
from .reconstruct_decoder import T5Stack
# from .reconstruct_decoder_save import T5Stack

from typing import Optional, Dict, List

from .arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
# from .reconstruct_decoder import T5LayerForDecoder
from .quantizer import Quantize, GBQuantize

from IPython import embed

logger = logging.getLogger(__name__)


@dataclass
class DenseOutput(ModelOutput):
    semantic_reps: Tensor = None
    semantic_reps2: Tensor = None
    loss: Tensor = None
    recon_loss: Tensor = None
    contrastive_loss: Tensor = None
    logits: Tensor = None
    label: Tensor = None
    code_id: Tensor = None


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    # token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SemanticIDLM(nn.Module):
    '''
    This is the model where no quantization codebook is added. Semantic Encoder -> Decoder.
    '''
    def __init__(
            self,
            lm: PreTrainedModel,
            recon_decoder: T5Stack,
            code_book: Quantize,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm = lm

        if "T5" in type(self.lm).__name__:
            self.decoder_embeddings = self.lm.shared
        elif "MPNet" in type(self.lm).__name__:
            self.decoder_embeddings = self.lm.mpnet.embeddings
        else:
            raise ValueError('Wrong PLM name !!!')

        self.recon_decoder = recon_decoder

        self.code_book = code_book 

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args


    def forward(
            self,
            encoder_batch,
            decoder_batch
    ):

        _, encoder_reps = self.encode(encoder_batch)
        _, encoder_reps2 = self.encode(encoder_batch)

        decoder_embedding_output = self.decoder_embeddings(decoder_batch['decoder_input_ids'])
        # decoder_embedding_output= torch.randn_like(decoder_embedding_output) ###################
        # decoder_embedding_output= torch.zeros_like(decoder_embedding_output) ###################

        decoder_input_hiddens = torch.cat([encoder_reps.unsqueeze(1), decoder_embedding_output], dim=1)

        bz, hz = encoder_reps.shape
        seq_len = decoder_input_hiddens.shape[1]
        # make sure that encoder_reps has the similar shape with decoder_input_hiddens
        hiddens = self.recon_decoder(query_embed=encoder_reps.unsqueeze(1).expand(bz, seq_len, hz), inputs_embeds=decoder_input_hiddens, attention_mask=decoder_batch['decoder_attention_mask']).last_hidden_state

        lm_logits, recon_loss = self.mlm_loss(hiddens, decoder_batch['decoder_labels'])

        # contrastive loss
        scores = torch.matmul(encoder_reps, encoder_reps2.transpose(0, 1))
        target = torch.arange(
            scores.size(0),
            device=scores.device,
            dtype=torch.long
        )

        if self.model_args.contrastive_obj:
            contrastive_loss = self.cross_entropy(scores, target)
        else:
            contrastive_loss = torch.FloatTensor([0]).squeeze(-1).to(recon_loss.device)

        return DenseOutput(
            loss=recon_loss + contrastive_loss,
            recon_loss=recon_loss,
            contrastive_loss=contrastive_loss,
            semantic_reps=encoder_reps,
            semantic_reps2=encoder_reps2,
            logits=torch.argmax(lm_logits, -1),
            label=decoder_batch['decoder_labels']
        )

    def encode(self, psg):
        if psg is None:
            return None, None
        psg = BatchEncoding(psg)

        if "T5" in type(self.lm).__name__:
            decoder_input_ids = torch.zeros((psg.encoder_input_ids.shape[0], 1), dtype=torch.long).to(psg.encoder_input_ids.device)            
            items_out = self.lm(psg['encoder_input_ids'], psg['encoder_attention_mask'], decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True)
            # hidden = items_out.last_hidden_state
            hidden = items_out.decoder_hidden_states[-1]
            reps = hidden[:, 0, :]
        elif "MPNet" in type(self.lm).__name__:
            items_out = self.lm(psg['encoder_input_ids'], psg['encoder_attention_mask'], return_dict=True, output_hidden_states=True)
            # hidden = items_out.last_hidden_state
            # hidden = items_out.decoder_hidden_states[-1]
            # reps = hidden[:, 0, :]
            hidden = items_out.hidden_states[-1]
            reps = mean_pooling(hidden, psg['encoder_attention_mask'])
        else:
            raise ValueError('Wrong PLM name !!!')

        return hidden, reps

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load model
        # lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if model_args.encoder_type == 'seq2seq':
            lm = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            hidden_size = lm.config.d_model
        elif model_args.encoder_type == 'enconly':
            lm = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            hidden_size = lm.config.hidden_size
        else:
            raise ValueError('Wrong encoder type!')

        # init reconstructor
        # reconstruction decoder config
        # assert model_args.recon_decoder_num_layer == 1
        # recon_decoder_config = copy.deepcopy(lm.config)
        recon_decoder_config = AutoConfig.from_pretrained('t5-base')
        recon_decoder_config.is_decoder = False
        recon_decoder_config.use_cache = False
        recon_decoder_config.is_encoder_decoder = False
        recon_decoder_config.num_layers = model_args.recon_decoder_num_layer
        recon_decoder = T5Stack(recon_decoder_config)

        reconstructor_path = os.path.join(model_args.model_name_or_path, 'reconstructor.pt')
        if os.path.exists(reconstructor_path):
            recon_decoder.load_state_dict(torch.load(reconstructor_path, map_location='cpu'))

        # init codebook
        # code_book = Quantize(hidden_size, model_args.quantization_pool_size)
        if model_args.gumbel_softmax:
            code_book = GBQuantize(hidden_size, model_args.quantization_pool_size)
        else:
            code_book = Quantize(hidden_size, model_args.quantization_pool_size)

        code_book_path = os.path.join(model_args.model_name_or_path, 'code_book.pt')
        if model_args.load_codebook:
            print('#############')
            print(f'You are loading in the kmeans center embeddings to the codebook from {model_args.load_codebook_dir}.')
            print('#############')
            readin = np.load(open(os.path.join(model_args.load_codebook_dir), 'rb'))
            code_book.embed.data = torch.FloatTensor(readin)
        elif os.path.exists(code_book_path):
            code_book.load_state_dict(torch.load(code_book_path, map_location='cpu'))
        else:
            print('#############')
            print('Random initializing the codebook embeddings.')
            print('#############')

        model = cls(
            lm=lm,
            recon_decoder=recon_decoder,
            code_book=code_book,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model

    def save(self, output_dir: str):
        # save lm
        self.lm.save_pretrained(output_dir)

        # save reconstructor
        torch.save(self.recon_decoder.state_dict(), os.path.join(output_dir, 'reconstructor.pt'))

        # save codebook
        # torch.save(self.code_book.state_dict(), os.path.join(output_dir, 'code_book.pt'))

    def mlm_loss(self, hiddens, labels):
        lm_logits = self.lm.lm_head(hiddens)
        masked_lm_loss = self.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            labels.view(-1)
        )
        return lm_logits, masked_lm_loss


class SemanticIDQELM(nn.Module):
    '''
    This is the model where the encoder can be optimized by the reconstruction loss and commitment loss,
    decoder can be trained by the reconstruction loss,
    while the codebook embedding is updated with exponential moving averages (EMA).

    The encoder is also optimized by the self-contrastive loss.
    '''
    def __init__(
            self,
            lm: PreTrainedModel,
            recon_decoder: T5Stack,
            code_book: Quantize,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm = lm

        if "T5" in type(self.lm).__name__:
            self.decoder_embeddings = self.lm.shared
        elif "MPNet" in type(self.lm).__name__:
            self.decoder_embeddings = self.lm.mpnet.embeddings
        else:
            raise ValueError('Wrong PLM name !!!')

        self.recon_decoder = recon_decoder

        self.code_book = code_book 

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args


    def forward(
            self,
            encoder_batch,
            decoder_batch
    ):

        _, encoder_reps = self.encode(encoder_batch)
        _, encoder_reps2 = self.encode(encoder_batch)

        quantized_reps, _, code_id = self.code_book(encoder_reps)

        decoder_embedding_output = self.decoder_embeddings(decoder_batch['decoder_input_ids'])
        decoder_input_hiddens = torch.cat([quantized_reps.unsqueeze(1), decoder_embedding_output], dim=1)

        bz, hz = encoder_reps.shape
        seq_len = decoder_input_hiddens.shape[1]
        # make sure that encoder_reps has the similar shape with decoder_input_hiddens
        hiddens = self.recon_decoder(query_embed=quantized_reps.unsqueeze(1).expand(bz, seq_len, hz), inputs_embeds=decoder_input_hiddens, attention_mask=decoder_batch['decoder_attention_mask']).last_hidden_state

        # reconstruction loss
        lm_logits, recon_loss = self.mlm_loss(hiddens, decoder_batch['decoder_labels'])

        # contrastive loss
        scores = torch.matmul(encoder_reps, encoder_reps2.transpose(0, 1))
        target = torch.arange(
            scores.size(0),
            device=scores.device,
            dtype=torch.long
        )

        if self.model_args.contrastive_obj:
            contrastive_loss = self.cross_entropy(scores, target)
        else:
            contrastive_loss = torch.FloatTensor([0]).squeeze(-1).to(recon_loss.device)

        # contrastive_loss = self.cross_entropy(scores, target)

        return DenseOutput(
            loss=contrastive_loss + recon_loss,
            recon_loss=recon_loss,
            contrastive_loss=contrastive_loss,
            semantic_reps=encoder_reps,
            semantic_reps2=encoder_reps2,
            code_id=code_id,
            logits = torch.argmax(lm_logits, -1),
            label=decoder_batch['decoder_labels']
        )

    def encode(self, psg):
        if psg is None:
            return None, None
        psg = BatchEncoding(psg)

        if "T5" in type(self.lm).__name__:
            decoder_input_ids = torch.zeros((psg.encoder_input_ids.shape[0], 1), dtype=torch.long).to(psg.encoder_input_ids.device)            
            items_out = self.lm(psg['encoder_input_ids'], psg['encoder_attention_mask'], decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True)
            # hidden = items_out.last_hidden_state
            hidden = items_out.decoder_hidden_states[-1]
            reps = hidden[:, 0, :]
        elif "MPNet" in type(self.lm).__name__:
            items_out = self.lm(psg['encoder_input_ids'], psg['encoder_attention_mask'], return_dict=True, output_hidden_states=True)
            # hidden = items_out.last_hidden_state
            # hidden = items_out.decoder_hidden_states[-1]
            # reps = hidden[:, 0, :]
            hidden = items_out.hidden_states[-1]
            reps = mean_pooling(hidden, psg['encoder_attention_mask'])
        else:
            raise ValueError('Wrong PLM name !!!')

        return hidden, reps

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load model
        # lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if model_args.encoder_type == 'seq2seq':
            lm = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            hidden_size = lm.config.d_model
        elif model_args.encoder_type == 'enconly':
            lm = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            hidden_size = lm.config.hidden_size
        else:
            raise ValueError('Wrong encoder type!')

        # init reconstructor
        # reconstruction decoder config
        # assert model_args.recon_decoder_num_layer == 1
        # recon_decoder_config = copy.deepcopy(lm.config)
        recon_decoder_config = AutoConfig.from_pretrained('t5-base')
        recon_decoder_config.is_decoder = False
        recon_decoder_config.use_cache = False
        recon_decoder_config.is_encoder_decoder = False
        recon_decoder_config.num_layers = model_args.recon_decoder_num_layer
        recon_decoder = T5Stack(recon_decoder_config)

        reconstructor_path = os.path.join(model_args.model_name_or_path, 'reconstructor.pt')
        if os.path.exists(reconstructor_path):
            recon_decoder.load_state_dict(torch.load(reconstructor_path, map_location='cpu'))

        # init codebook
        code_book = GBQuantize(hidden_size, model_args.quantization_pool_size)
        # code_book = Quantize(hidden_size, model_args.quantization_pool_size)
        # if model_args.gumbel_softmax:
        #     code_book = GBQuantize(hidden_size, model_args.quantization_pool_size)
        # else:
        #     code_book = Quantize(hidden_size, model_args.quantization_pool_size)

        code_book_path = os.path.join(model_args.model_name_or_path, 'code_book.pt')
        if model_args.load_codebook:
            print('#############')
            print(f'You are loading in the kmeans center embeddings to the codebook from {model_args.load_codebook_dir}.')
            print('#############')
            readin = np.load(open(os.path.join(model_args.load_codebook_dir), 'rb'))
            code_book.embed.data = torch.FloatTensor(readin)
        elif os.path.exists(code_book_path):
            code_book.load_state_dict(torch.load(code_book_path, map_location='cpu'))
        else:
            print('#############')
            print('Random initializing the codebook embeddings.')
            print('#############')

        model = cls(
            lm=lm,
            recon_decoder=recon_decoder,
            code_book=code_book,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model

    def save(self, output_dir: str):
        # save lm
        self.lm.save_pretrained(output_dir)

        # save reconstructor
        torch.save(self.recon_decoder.state_dict(), os.path.join(output_dir, 'reconstructor.pt'))

        # save codebook
        torch.save(self.code_book.state_dict(), os.path.join(output_dir, 'code_book.pt'))

    def mlm_loss(self, hiddens, labels):
        # embed()
        lm_logits = self.lm.lm_head(hiddens)
        masked_lm_loss = self.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            labels.view(-1)
        )
        return lm_logits, masked_lm_loss



# class SemanticIDQLM(nn.Module):
#     '''
#     This is the model where only the codebook embeddings and decoder is trained while the encoder is not trained.
#     A more advanced model please refer to the following one below.
#     !!!!!! Not maintained !!!!!!
#     '''
#     def __init__(
#             self,
#             lm: PreTrainedModel,
#             model_args: ModelArguments = None,
#             data_args: DataArguments = None,
#             train_args: TrainingArguments = None,
#     ):
#         super().__init__()

#         self.lm = lm

#         self.decoder_embeddings = self.lm.shared

#         # reconstruction decoder config
#         recon_decoder_config = copy.deepcopy(lm.config)
#         recon_decoder_config.is_decoder = False
#         recon_decoder_config.use_cache = False
#         recon_decoder_config.is_encoder_decoder = False
#         recon_decoder_config.num_layers = model_args.recon_decoder_num_layer

#         self.recon_decoder = T5Stack(recon_decoder_config)
#         # self.recon_decoder.apply(self.lm._init_weights)

#         self.code_book = nn.Parameter(torch.FloatTensor(model_args.quantization_pool_size, lm.config.d_model))
#         nn.init.xavier_normal_(self.code_book)

#         self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

#         self.model_args = model_args
#         self.train_args = train_args
#         self.data_args = data_args


#     def forward(
#             self,
#             encoder_batch,
#             decoder_batch
#     ):

#         encoder_hidden, encoder_reps = self.encode(encoder_batch)
#         quantized_id = torch.argmax(torch.mm(encoder_reps, self.code_book.t()),1)
#         quantized_reps = self.code_book[quantized_id]

#         decoder_embedding_output = self.decoder_embeddings(decoder_batch['decoder_input_ids'])
#         decoder_input_hiddens = torch.cat([quantized_reps.unsqueeze(1), decoder_embedding_output], dim=1)

#         bz, hz = encoder_reps.shape
#         seq_len = decoder_input_hiddens.shape[1]
#         # make sure that encoder_reps has the similar shape with decoder_input_hiddens
#         hiddens = self.recon_decoder(query_embed=quantized_reps.unsqueeze(1).expand(bz, seq_len, hz), inputs_embeds=decoder_input_hiddens, attention_mask=decoder_batch['decoder_attention_mask']).last_hidden_state

#         lm_logits, loss = self.mlm_loss(hiddens, decoder_batch['decoder_labels'])

#         # embed()

#         return DenseOutput(
#             loss=loss,
#             semantic_reps=encoder_reps,
#             logits = torch.argmax(lm_logits, -1),
#             label=decoder_batch['decoder_labels']
#         )

#     def encode(self, psg):
#         if psg is None:
#             return None, None
#         psg = BatchEncoding(psg)

#         if "T5" in type(self.lm).__name__:
#             decoder_input_ids = torch.zeros((psg.encoder_input_ids.shape[0], 1), dtype=torch.long).to(psg.encoder_input_ids.device)            
#             items_out = self.lm(psg['encoder_input_ids'], psg['encoder_attention_mask'], decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True)
#             # hidden = items_out.last_hidden_state
#             hidden = items_out.decoder_hidden_states[-1]
#             reps = hidden[:, 0, :]
#         else:
#             raise ValueError('Wrong PLM name !!!')

#         return hidden, reps

#     @classmethod
#     def build(
#             cls,
#             model_args: ModelArguments,
#             data_args: DataArguments,
#             train_args: TrainingArguments,
#             **hf_kwargs,
#     ):
#         # load model
#         # lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
#         lm = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)

#         model = cls(
#             lm=lm,
#             model_args=model_args,
#             data_args=data_args,
#             train_args=train_args,
#         )
#         return model

#     def save(self, output_dir: str):
#         self.lm.save_pretrained(output_dir)

#     def mlm_loss(self, hiddens, labels):
#         # embed()
#         lm_logits = self.lm.lm_head(hiddens)
#         masked_lm_loss = self.cross_entropy(
#             lm_logits.view(-1, lm_logits.size(-1)),
#             labels.view(-1)
#         )
#         return lm_logits, masked_lm_loss


# class SemanticIDQELM(nn.Module):
#     '''
#     This is the model where the encoder can be optimized by the reconstruction loss and commitment loss,
#     decoder can be trained by the reconstruction loss,
#     while the codebook embedding is updated with exponential moving averages (EMA).
#     '''
#     def __init__(
#             self,
#             lm: PreTrainedModel,
#             recon_decoder: T5Stack,
#             code_book: Quantize,
#             model_args: ModelArguments = None,
#             data_args: DataArguments = None,
#             train_args: TrainingArguments = None,
#     ):
#         super().__init__()

#         self.lm = lm

#         if "T5" in type(self.lm).__name__:
#             self.decoder_embeddings = self.lm.shared
#         elif "MPNet" in type(self.lm).__name__:
#             self.decoder_embeddings = self.lm.mpnet.embeddings
#         else:
#             raise ValueError('Wrong PLM name !!!')

#         self.recon_decoder = recon_decoder

#         self.code_book = code_book 

#         self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

#         self.model_args = model_args
#         self.train_args = train_args
#         self.data_args = data_args


#     def forward(
#             self,
#             encoder_batch,
#             decoder_batch
#     ):

#         encoder_hidden, encoder_reps = self.encode(encoder_batch)

#         quantized_reps, commit_loss, code_id = self.code_book(encoder_reps)

#         decoder_embedding_output = self.decoder_embeddings(decoder_batch['decoder_input_ids'])
#         decoder_input_hiddens = torch.cat([quantized_reps.unsqueeze(1), decoder_embedding_output], dim=1)

#         bz, hz = encoder_reps.shape
#         seq_len = decoder_input_hiddens.shape[1]
#         # make sure that encoder_reps has the similar shape with decoder_input_hiddens
#         hiddens = self.recon_decoder(query_embed=quantized_reps.unsqueeze(1).expand(bz, seq_len, hz), inputs_embeds=decoder_input_hiddens, attention_mask=decoder_batch['decoder_attention_mask']).last_hidden_state

#         lm_logits, loss = self.mlm_loss(hiddens, decoder_batch['decoder_labels'])

#         return DenseOutput(
#             loss=loss + commit_loss * self.model_args.commitment_loss_weight,
#             semantic_reps=encoder_reps,
#             code_id=code_id,
#             logits = torch.argmax(lm_logits, -1),
#             label=decoder_batch['decoder_labels']
#         )

#     def encode(self, psg):
#         if psg is None:
#             return None, None
#         psg = BatchEncoding(psg)

#         if "T5" in type(self.lm).__name__:
#             decoder_input_ids = torch.zeros((psg.encoder_input_ids.shape[0], 1), dtype=torch.long).to(psg.encoder_input_ids.device)            
#             items_out = self.lm(psg['encoder_input_ids'], psg['encoder_attention_mask'], decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True)
#             # hidden = items_out.last_hidden_state
#             hidden = items_out.decoder_hidden_states[-1]
#             reps = hidden[:, 0, :]
#         elif "MPNet" in type(self.lm).__name__:
#             items_out = self.lm(psg['encoder_input_ids'], psg['encoder_attention_mask'], return_dict=True, output_hidden_states=True)
#             # hidden = items_out.last_hidden_state
#             # hidden = items_out.decoder_hidden_states[-1]
#             # reps = hidden[:, 0, :]
#             hidden = items_out.hidden_states[-1]
#             reps = mean_pooling(hidden, psg['encoder_attention_mask'])
#         else:
#             raise ValueError('Wrong PLM name !!!')

#         return hidden, reps

#     @classmethod
#     def build(
#             cls,
#             model_args: ModelArguments,
#             data_args: DataArguments,
#             train_args: TrainingArguments,
#             **hf_kwargs,
#     ):
#         # load model
#         # lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
#         if model_args.encoder_type == 'seq2seq':
#             lm = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
#             hidden_size = lm.config.d_model
#         elif model_args.encoder_type == 'enconly':
#             lm = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
#             hidden_size = lm.config.hidden_size
#         else:
#             raise ValueError('Wrong encoder type!')

#         # init reconstructor
#         # reconstruction decoder config
#         assert model_args.recon_decoder_num_layer == 1
#         # recon_decoder_config = copy.deepcopy(lm.config)
#         recon_decoder_config = AutoConfig.from_pretrained('t5-base')
#         recon_decoder_config.is_decoder = False
#         recon_decoder_config.use_cache = False
#         recon_decoder_config.is_encoder_decoder = False
#         recon_decoder_config.num_layers = model_args.recon_decoder_num_layer
#         recon_decoder = T5Stack(recon_decoder_config)

#         reconstructor_path = os.path.join(model_args.model_name_or_path, 'reconstructor.pt')
#         if os.path.exists(reconstructor_path):
#             recon_decoder.load_state_dict(torch.load(reconstructor_path, map_location='cpu'))

#         # init codebook
#         # code_book = Quantize(hidden_size, model_args.quantization_pool_size)
#         if model_args.gumbel_softmax:
#             code_book = GBQuantize(hidden_size, model_args.quantization_pool_size)
#         else:
#             code_book = Quantize(hidden_size, model_args.quantization_pool_size)

#         code_book_path = os.path.join(model_args.model_name_or_path, 'code_book.pt')
#         if model_args.load_codebook:
#             print('#############')
#             print(f'You are loading in the kmeans center embeddings to the codebook from {model_args.load_codebook_dir}.')
#             print('#############')
#             readin = np.load(open(os.path.join(model_args.load_codebook_dir), 'rb'))
#             code_book.embed.data = torch.FloatTensor(readin)
#         elif os.path.exists(code_book_path):
#             code_book.load_state_dict(torch.load(code_book_path, map_location='cpu'))
#         else:
#             print('#############')
#             print('Random initializing the codebook embeddings.')
#             print('#############')

#         model = cls(
#             lm=lm,
#             recon_decoder=recon_decoder,
#             code_book=code_book,
#             model_args=model_args,
#             data_args=data_args,
#             train_args=train_args,
#         )
#         return model

#     def save(self, output_dir: str):
#         # save lm
#         self.lm.save_pretrained(output_dir)

#         # save reconstructor
#         torch.save(self.recon_decoder.state_dict(), os.path.join(output_dir, 'reconstructor.pt'))

#         # save codebook
#         torch.save(self.code_book.state_dict(), os.path.join(output_dir, 'code_book.pt'))

#     def mlm_loss(self, hiddens, labels):
#         # embed()
#         lm_logits = self.lm.lm_head(hiddens)
#         masked_lm_loss = self.cross_entropy(
#             lm_logits.view(-1, lm_logits.size(-1)),
#             labels.view(-1)
#         )
#         return lm_logits, masked_lm_loss


# class SemanticIDQECLM(nn.Module):
#     '''
#     This is the model where the encoder can be optimized by the reconstruction loss and commitment loss,
#     decoder can be trained by the reconstruction loss,
#     while the codebook embedding is updated with exponential moving averages (EMA).

#     The encoder is also optimized by the self-contrastive loss.
#     '''
#     def __init__(
#             self,
#             lm: PreTrainedModel,
#             recon_decoder: T5Stack,
#             code_book: Quantize,
#             model_args: ModelArguments = None,
#             data_args: DataArguments = None,
#             train_args: TrainingArguments = None,
#     ):
#         super().__init__()

#         self.lm = lm

#         if "T5" in type(self.lm).__name__:
#             self.decoder_embeddings = self.lm.shared
#         elif "MPNet" in type(self.lm).__name__:
#             self.decoder_embeddings = self.lm.mpnet.embeddings
#         else:
#             raise ValueError('Wrong PLM name !!!')

#         self.recon_decoder = recon_decoder

#         self.code_book = code_book 

#         self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

#         self.model_args = model_args
#         self.train_args = train_args
#         self.data_args = data_args


#     def forward(
#             self,
#             encoder_batch,
#             decoder_batch
#     ):

#         _, encoder_reps = self.encode(encoder_batch)
#         _, encoder_reps2 = self.encode(encoder_batch)

#         quantized_reps, commit_loss, code_id = self.code_book(encoder_reps)

#         decoder_embedding_output = self.decoder_embeddings(decoder_batch['decoder_input_ids'])
#         decoder_input_hiddens = torch.cat([quantized_reps.unsqueeze(1), decoder_embedding_output], dim=1)

#         bz, hz = encoder_reps.shape
#         seq_len = decoder_input_hiddens.shape[1]
#         # make sure that encoder_reps has the similar shape with decoder_input_hiddens
#         hiddens = self.recon_decoder(query_embed=quantized_reps.unsqueeze(1).expand(bz, seq_len, hz), inputs_embeds=decoder_input_hiddens, attention_mask=decoder_batch['decoder_attention_mask']).last_hidden_state

#         # reconstruction loss
#         lm_logits, recon_loss = self.mlm_loss(hiddens, decoder_batch['decoder_labels'])

#         # contrastive loss
#         scores = torch.matmul(encoder_reps, encoder_reps2.transpose(0, 1))
#         target = torch.arange(
#             scores.size(0),
#             device=scores.device,
#             dtype=torch.long
#         )

#         contrastive_loss = self.cross_entropy(scores, target)

#         return DenseOutput(
#             loss=contrastive_loss + recon_loss + commit_loss * self.model_args.commitment_loss_weight,
#             semantic_reps=encoder_reps,
#             semantic_reps2=encoder_reps2,
#             code_id=code_id,
#             logits = torch.argmax(lm_logits, -1),
#             label=decoder_batch['decoder_labels']
#         )

#     def encode(self, psg):
#         if psg is None:
#             return None, None
#         psg = BatchEncoding(psg)

#         if "T5" in type(self.lm).__name__:
#             decoder_input_ids = torch.zeros((psg.encoder_input_ids.shape[0], 1), dtype=torch.long).to(psg.encoder_input_ids.device)            
#             items_out = self.lm(psg['encoder_input_ids'], psg['encoder_attention_mask'], decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True)
#             # hidden = items_out.last_hidden_state
#             hidden = items_out.decoder_hidden_states[-1]
#             reps = hidden[:, 0, :]
#         elif "MPNet" in type(self.lm).__name__:
#             items_out = self.lm(psg['encoder_input_ids'], psg['encoder_attention_mask'], return_dict=True, output_hidden_states=True)
#             # hidden = items_out.last_hidden_state
#             # hidden = items_out.decoder_hidden_states[-1]
#             # reps = hidden[:, 0, :]
#             hidden = items_out.hidden_states[-1]
#             reps = mean_pooling(hidden, psg['encoder_attention_mask'])
#         else:
#             raise ValueError('Wrong PLM name !!!')

#         return hidden, reps

#     @classmethod
#     def build(
#             cls,
#             model_args: ModelArguments,
#             data_args: DataArguments,
#             train_args: TrainingArguments,
#             **hf_kwargs,
#     ):
#         # load model
#         # lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
#         if model_args.encoder_type == 'seq2seq':
#             lm = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
#             hidden_size = lm.config.d_model
#         elif model_args.encoder_type == 'enconly':
#             lm = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
#             hidden_size = lm.config.hidden_size
#         else:
#             raise ValueError('Wrong encoder type!')

#         # init reconstructor
#         # reconstruction decoder config
#         assert model_args.recon_decoder_num_layer == 1
#         # recon_decoder_config = copy.deepcopy(lm.config)
#         recon_decoder_config = AutoConfig.from_pretrained('t5-base')
#         recon_decoder_config.is_decoder = False
#         recon_decoder_config.use_cache = False
#         recon_decoder_config.is_encoder_decoder = False
#         recon_decoder_config.num_layers = model_args.recon_decoder_num_layer
#         recon_decoder = T5Stack(recon_decoder_config)

#         reconstructor_path = os.path.join(model_args.model_name_or_path, 'reconstructor.pt')
#         if os.path.exists(reconstructor_path):
#             recon_decoder.load_state_dict(torch.load(reconstructor_path, map_location='cpu'))

#         # init codebook
#         # code_book = Quantize(hidden_size, model_args.quantization_pool_size)
#         if model_args.gumbel_softmax:
#             code_book = GBQuantize(hidden_size, model_args.quantization_pool_size)
#         else:
#             code_book = Quantize(hidden_size, model_args.quantization_pool_size)

#         code_book_path = os.path.join(model_args.model_name_or_path, 'code_book.pt')
#         if model_args.load_codebook:
#             print('#############')
#             print(f'You are loading in the kmeans center embeddings to the codebook from {model_args.load_codebook_dir}.')
#             print('#############')
#             readin = np.load(open(os.path.join(model_args.load_codebook_dir), 'rb'))
#             code_book.embed.data = torch.FloatTensor(readin)
#         elif os.path.exists(code_book_path):
#             code_book.load_state_dict(torch.load(code_book_path, map_location='cpu'))
#         else:
#             print('#############')
#             print('Random initializing the codebook embeddings.')
#             print('#############')

#         model = cls(
#             lm=lm,
#             recon_decoder=recon_decoder,
#             code_book=code_book,
#             model_args=model_args,
#             data_args=data_args,
#             train_args=train_args,
#         )
#         return model

#     def save(self, output_dir: str):
#         # save lm
#         self.lm.save_pretrained(output_dir)

#         # save reconstructor
#         torch.save(self.recon_decoder.state_dict(), os.path.join(output_dir, 'reconstructor.pt'))

#         # save codebook
#         torch.save(self.code_book.state_dict(), os.path.join(output_dir, 'code_book.pt'))

#     def mlm_loss(self, hiddens, labels):
#         # embed()
#         lm_logits = self.lm.lm_head(hiddens)
#         masked_lm_loss = self.cross_entropy(
#             lm_logits.view(-1, lm_logits.size(-1)),
#             labels.view(-1)
#         )
#         return lm_logits, masked_lm_loss


# class SemanticIDCLM(nn.Module):
#     '''
#     The encoder is optimized by the self-contrastive loss.
#     '''
#     def __init__(
#             self,
#             lm: PreTrainedModel,
#             recon_decoder: T5Stack,
#             code_book: Quantize,
#             model_args: ModelArguments = None,
#             data_args: DataArguments = None,
#             train_args: TrainingArguments = None,
#     ):
#         super().__init__()

#         self.lm = lm

#         if "T5" in type(self.lm).__name__:
#             self.decoder_embeddings = self.lm.shared
#         elif "MPNet" in type(self.lm).__name__:
#             self.decoder_embeddings = self.lm.mpnet.embeddings
#         else:
#             raise ValueError('Wrong PLM name !!!')

#         self.recon_decoder = recon_decoder

#         self.code_book = code_book 

#         self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

#         self.model_args = model_args
#         self.train_args = train_args
#         self.data_args = data_args


#     def forward(
#             self,
#             encoder_batch,
#             decoder_batch
#     ):

#         _, encoder_reps = self.encode(encoder_batch)
#         _, encoder_reps2 = self.encode(encoder_batch)

#         quantized_reps, commit_loss, code_id = self.code_book(encoder_reps)

#         decoder_embedding_output = self.decoder_embeddings(decoder_batch['decoder_input_ids'])
#         decoder_input_hiddens = torch.cat([quantized_reps.unsqueeze(1), decoder_embedding_output], dim=1)

#         bz, hz = encoder_reps.shape
#         seq_len = decoder_input_hiddens.shape[1]
#         # make sure that encoder_reps has the similar shape with decoder_input_hiddens
#         hiddens = self.recon_decoder(query_embed=quantized_reps.unsqueeze(1).expand(bz, seq_len, hz), inputs_embeds=decoder_input_hiddens, attention_mask=decoder_batch['decoder_attention_mask']).last_hidden_state

#         # reconstruction loss
#         lm_logits, recon_loss = self.mlm_loss(hiddens, decoder_batch['decoder_labels'])

#         # contrastive loss
#         scores = torch.matmul(encoder_reps, encoder_reps2.transpose(0, 1))
#         target = torch.arange(
#             scores.size(0),
#             device=scores.device,
#             dtype=torch.long
#         )

#         contrastive_loss = self.cross_entropy(scores, target)

#         return DenseOutput(
#             loss=contrastive_loss,
#             semantic_reps=encoder_reps,
#             semantic_reps2=encoder_reps2,
#             code_id=code_id,
#             logits = torch.argmax(lm_logits, -1),
#             label=decoder_batch['decoder_labels']
#         )

#     def encode(self, psg):
#         if psg is None:
#             return None, None
#         psg = BatchEncoding(psg)

#         if "T5" in type(self.lm).__name__:
#             decoder_input_ids = torch.zeros((psg.encoder_input_ids.shape[0], 1), dtype=torch.long).to(psg.encoder_input_ids.device)            
#             items_out = self.lm(psg['encoder_input_ids'], psg['encoder_attention_mask'], decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True)
#             # hidden = items_out.last_hidden_state
#             hidden = items_out.decoder_hidden_states[-1]
#             reps = hidden[:, 0, :]
#         elif "MPNet" in type(self.lm).__name__:
#             items_out = self.lm(psg['encoder_input_ids'], psg['encoder_attention_mask'], return_dict=True, output_hidden_states=True)
#             # hidden = items_out.last_hidden_state
#             # hidden = items_out.decoder_hidden_states[-1]
#             # reps = hidden[:, 0, :]
#             hidden = items_out.hidden_states[-1]
#             reps = mean_pooling(hidden, psg['encoder_attention_mask'])
#         else:
#             raise ValueError('Wrong PLM name !!!')

#         return hidden, reps

#     @classmethod
#     def build(
#             cls,
#             model_args: ModelArguments,
#             data_args: DataArguments,
#             train_args: TrainingArguments,
#             **hf_kwargs,
#     ):
#         # load model
#         # lm = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
#         if model_args.encoder_type == 'seq2seq':
#             lm = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
#             hidden_size = lm.config.d_model
#         elif model_args.encoder_type == 'enconly':
#             lm = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
#             hidden_size = lm.config.hidden_size
#         else:
#             raise ValueError('Wrong encoder type!')

#         # init reconstructor
#         # reconstruction decoder config
#         assert model_args.recon_decoder_num_layer == 1
#         # recon_decoder_config = copy.deepcopy(lm.config)
#         recon_decoder_config = AutoConfig.from_pretrained('t5-base')
#         recon_decoder_config.is_decoder = False
#         recon_decoder_config.use_cache = False
#         recon_decoder_config.is_encoder_decoder = False
#         recon_decoder_config.num_layers = model_args.recon_decoder_num_layer
#         recon_decoder = T5Stack(recon_decoder_config)

#         reconstructor_path = os.path.join(model_args.model_name_or_path, 'reconstructor.pt')
#         if os.path.exists(reconstructor_path):
#             recon_decoder.load_state_dict(torch.load(reconstructor_path, map_location='cpu'))

#         # init codebook
#         # code_book = Quantize(hidden_size, model_args.quantization_pool_size)
#         if model_args.gumbel_softmax:
#             code_book = GBQuantize(hidden_size, model_args.quantization_pool_size)
#         else:
#             code_book = Quantize(hidden_size, model_args.quantization_pool_size)

#         code_book_path = os.path.join(model_args.model_name_or_path, 'code_book.pt')
#         if os.path.exists(code_book_path):
#             code_book.load_state_dict(torch.load(code_book_path, map_location='cpu'))
#         elif model_args.load_codebook:
#             print('#############')
#             print(f'You are loading in the kmeans center embeddings to the codebook from {model_args.load_codebook_dir}.')
#             print('#############')
#             readin = np.load(open(os.path.join(model_args.load_codebook_dir), 'rb')).transpose()
#             code_book.embed.data = torch.FloatTensor(readin)
#         else:
#             print('#############')
#             print('Random initializing the codebook embeddings.')
#             print('#############')

#         model = cls(
#             lm=lm,
#             recon_decoder=recon_decoder,
#             code_book=code_book,
#             model_args=model_args,
#             data_args=data_args,
#             train_args=train_args,
#         )
#         return model

#     def save(self, output_dir: str):
#         # save lm
#         self.lm.save_pretrained(output_dir)

#         # save reconstructor
#         torch.save(self.recon_decoder.state_dict(), os.path.join(output_dir, 'reconstructor.pt'))

#         # save codebook
#         torch.save(self.code_book.state_dict(), os.path.join(output_dir, 'code_book.pt'))

#     def mlm_loss(self, hiddens, labels):
#         # embed()
#         lm_logits = self.lm.lm_head(hiddens)
#         masked_lm_loss = self.cross_entropy(
#             lm_logits.view(-1, lm_logits.size(-1)),
#             labels.view(-1)
#         )
#         return lm_logits, masked_lm_loss
