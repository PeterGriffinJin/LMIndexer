# This code script is for generating semantic ids for document in a given corpus.

DOMAIN=beauty
PROCESSED_DIR=~/quic-efs/user/bowenjin/SemanticID/data/$DOMAIN

# CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512   # this is for amazon datasets
# CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-4/0.7/quantization-512   # This is for triviaqa
# CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-5120   # This is for NQ
# CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-51200   # This is for NQ
CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-128


SAVE_ID_DIR=$CHECKPOINT_DIR/semanticid.txt

MODEL='t5-base'      #   sentence-transformers/all-mpnet-base-v2

match=$(echo $MODEL | grep "/")
if [[ "$match"=="$MODEL" ]]
then
    MODEL_SIMPLE_NAME=$(echo $MODEL | cut -d "/" -f 2)
else
    MODEL_SIMPLE_NAME=$MODEL
fi

LR="1e-2"
MASK_RATIO=0

quan_size=128

# LOAD_CORPORA_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/2e-3/0.5/quantization-512
# LOAD_CORPORA_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/5e-3/0.5/quantization-5120
# LOAD_CORPORA_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-51200
LOAD_CORPORA_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-128


CUDA_VISIBLE_DEVICES=0 python3 -m IDGen.generate_code_seq  \
    --output_dir tmp/  \
    --model_name_or_path $CHECKPOINT_DIR  \
    --encoder_type seq2seq \
    --save_id_dir $SAVE_ID_DIR  \
    --do_eval  \
    --target_position 2 \
    --hn_num 0 \
    --add_quantization True \
    --gumbel_softmax True \
    --quantization_pool_size $quan_size \
    --train_path $LOAD_CORPORA_DIR/document.2.${MODEL_SIMPLE_NAME}.tokenized.json  \
    --eval_path $LOAD_CORPORA_DIR/document.2.${MODEL_SIMPLE_NAME}.tokenized.json  \
    --per_device_eval_batch_size 16 \
    --max_len 512  \
    --decoder_mlm_probability $MASK_RATIO \
    --evaluation_strategy steps \
    --report_to none \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 32
