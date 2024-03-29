import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

    # masked decoder input
    recon_decoder_num_layer: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of layers in reconstruction T5 decoder"
        }
    )

    # quantization information
    add_quantization: bool = field(
        default=False,
        metadata={
            "help": "Add quantization or not"
        }
    )
    quantization_pool_size: Optional[int] = field(
        default=256,
        metadata={
            "help": "The number of quantization vectors in the quantization pool"
        }
    )
    commitment_loss_weight: Optional[float] = field(
        default=0.25,
        metadata={
            "help": "The weight of the commitment loss."
        }
    )
    save_id_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The target file dir to save the semantic id results."
        }
    )
    load_codebook: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Load codebook from kmeans or not."
        }
    )
    load_codebook_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The kmeans codebook init file."
        }
    )
    encoder_type: Optional[str] = field(
        default="seq2seq",
        metadata={
            "help": "The type of the encoder, can be either seq2seq or enconly."
        }
    )
    contrastive_obj: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use contrastive loss to train the encoder or not."
        }
    )
    gumbel_softmax: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use gumbel softmax or not."
        }
    )
    fix_GB_codebook: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Fix the codebook embeddings or not when using GB."
        }
    )
    fix_reconstructor: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Fix the reconstructor parameters or not."
        }
    )
    save_embed_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The target file dir to save the encoder soft embedding results."
        }
    )
    target_position: Optional[int] = field(
        default=None,
        metadata={
            "help": "The target semantic ID position to learn."
        }
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to single train file"}
    )
    eval_path: str = field(
        default=None, metadata={"help": "Path to eval file"}
    )
    query_path: str = field(
        default=None, metadata={"help": "Path to query file"}
    )
    corpus_path: str = field(
        default=None, metadata={"help": "Path to corpus file"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "Path to data directory"}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the single data file"}
    )
    processed_data_path: str = field(
        default=None, metadata={"help": "Path to processed data directory"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    hn_num: int = field(
        default=4, metadata={"help": "number of negatives used"}
    )
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"}
    )

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    
    encode_is_qry: bool = field(default=False)
    save_trec: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    query_column_names: str = field(
        default="id,text",
        metadata={"help": "column names for the tsv data format"}
    )
    doc_column_names: str = field(
        default="id,title,text",
        metadata={"help": "column names for the tsv data format"}
    )

    # mlm pretrain
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    decoder_mlm_probability: Optional[float] = field(
        default=0.15,
        metadata={
            "help": "The probability of token to be masked/corrupted during Mask Language Modeling"
        }
    )

    # rerank
    pos_rerank_num: int = field(default=5)
    neg_rerank_num: int = field(default=15)

    # coarse-grained node classification
    class_num: int = field(default=10)

    def __post_init__(self):
        pass


@dataclass
class DenseTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)

    fix_bert: bool = field(default=False, metadata={"help": "fix BERT encoder during training or not"})

    mlm_loss: bool = field(default=False, metadata={"help": "use mlm loss or not"})
    mlm_weight: float = field(default=1, metadata={"help": "weight of mlm loss"})

    fix_semanticID_generator: bool = field(default=False, metadata={"help": "use mlm loss or not"})

@dataclass
class DenseEncodingArguments(TrainingArguments):
    use_gpu: bool = field(default=False, metadata={"help": "Use GPU for encoding"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    save_path: str = field(default=None, metadata={"help": "where to save the result file"})
    retrieve_domain: str = field(default=None, metadata={"help": "name of the retrieve domain"})
    source_domain: str = field(default=None, metadata={"help": "name of the source domain"})
