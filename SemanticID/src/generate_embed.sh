# This code script is for generating semantic ids for document in a given corpus.

DOMAIN=beauty
PROCESSED_DIR=~/quic-efs/user/bowenjin/SemanticID/data/$DOMAIN

# CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/beauty/position1/EMA/1e-3/0.5/quantization-512
# CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/beauty/EMA/1e-2/0/quantization-512
# CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/nocode/contrastive/1e-3
# CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/nocode/contrastive/1e-3/2
# CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/nocode/contrastive/1e-3
CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/nocode/contrastive/1e-3/quantization-128
SAVE_EMBED_DIR=$CHECKPOINT_DIR/embed.pt

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

CUDA_VISIBLE_DEVICES=7 python3 -m IDGen.generate_embed  \
    --output_dir tmp/  \
    --model_name_or_path $CHECKPOINT_DIR  \
    --encoder_type seq2seq \
    --save_embed_dir $SAVE_EMBED_DIR  \
    --do_eval  \
    --add_quantization False \
    --gumbel_softmax True \
    --quantization_pool_size $quan_size \
    --train_path $PROCESSED_DIR/document.${MODEL_SIMPLE_NAME}.tokenized.json  \
    --eval_path $PROCESSED_DIR/document.${MODEL_SIMPLE_NAME}.tokenized.json  \
    --per_device_eval_batch_size 256 \
    --max_len 128  \
    --decoder_mlm_probability $MASK_RATIO \
    --evaluation_strategy steps \
    --report_to none \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 32
