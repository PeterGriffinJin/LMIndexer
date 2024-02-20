# This code script is for encoder (whole T5) + decoder

import logging
import os
import sys
import wandb

from IDGen.arguments import DataArguments, ModelArguments
from IDGen.arguments import DenseTrainingArguments as TrainingArguments
from IDGen.data import TrainIDCollator, TrainIDDataset, EvalIDDataset
from IDGen.modeling import SemanticIDCLM
from IDGen.trainer import DenseIDTrainer
from IDGen.utils import calculate_metrics, calculate_metrics_nocode

import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import is_main_process


logger = logging.getLogger(__name__)

from IPython import embed

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank in (0, -1):
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

    if training_args.local_rank in (0, -1) and 'wandb' in training_args.report_to:  # only on main process
        # Initialize wandb run
        wandb.login()
        wandb.init(project="SemanticIDGen")

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    # print("#####################")
    # if model_args.add_quantization:
    #     if model_args.only_code:
    #         logger.info("Using the model: SemanticIDQLM")
    #         model_class = SemanticIDQLM
    #     else:
    #         if model_args.contrastive_obj:
    #             logger.info("Using the model: SemanticIDQECLM")
    #             model_class = SemanticIDQECLM
    #         else:
    #             logger.info("Using the model: SemanticIDQELM")
    #             model_class = SemanticIDQELM
    # else:
    #     logger.info("Using the model: SemanticIDLM")
    #     model_class = SemanticIDLM
    # print("#####################")

    model = SemanticIDCLM.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # fix parameters or not
    # raise ValueError('Be careful about if you want to fix the T5 id generator first')
    if training_args.fix_semanticID_generator:
        for param in model.lm.parameters():
            param.requires_grad = False

    if model_args.fix_reconstructor:
        for param in model.recon_decoder.parameters():
            param.requires_grad = False

    # train_dataset = TrainIDDataset(tokenizer, data_args, shuffle_seed=training_args.seed, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    train_dataset = TrainIDDataset(tokenizer, data_args, shuffle_seed=None, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    eval_dataset = EvalIDDataset(tokenizer, data_args, shuffle_seed=training_args.seed, cache_dir=data_args.data_cache_dir or model_args.cache_dir) if data_args.eval_path is not None else None

    trainer_cls = DenseIDTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TrainIDCollator(
            tokenizer,
            max_seq_length=data_args.max_len,
            mlm=False,
            decoder_mlm_probability=data_args.decoder_mlm_probability
        ),
        compute_metrics=calculate_metrics(codebook_size=model_args.quantization_pool_size) if model_args.add_quantization else calculate_metrics_nocode,
    )

    # if training_args.local_rank in (0, -1):
    #     embed()

    if model_args.gumbel_softmax:
        model.code_book.embed = model.code_book.embed.to(model.lm.device)
        if model_args.fix_GB_codebook:
            model.code_book.embed.requires_grad = False

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

    # if training_args.local_rank in (0, -1):
    #     embed()


if __name__ == "__main__":
    main()
