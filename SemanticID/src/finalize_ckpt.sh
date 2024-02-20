# This code script is for finalizing the model (loading the last codebook into the model embedding map).

DOMAIN=beauty
PROCESSED_DIR=~/quic-efs/user/bowenjin/SemanticID/data/$DOMAIN
LOG_DIR=~/quic-efs/user/bowenjin/SemanticID/logs/$DOMAIN
CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN

MODEL='t5-base'      #   sentence-transformers/all-mpnet-base-v2

LR="1e-2"
PRETRAIN_MASK_RATIO=1
MASK_RATIO=0.7

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

match=$(echo $MODEL | grep "/")
if [[ "$match"=="$MODEL" ]]
then
    MODEL_SIMPLE_NAME=$(echo $MODEL | cut -d "/" -f 2)
else
    MODEL_SIMPLE_NAME=$MODEL
fi

echo "start training..."

MODEL_TYPE='GB'
quan_size=128

# # load the pretrained decoder parameters
# LOAD_CKPT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512   # for amazon datasets
# LOAD_CKPT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-4/0.7/quantization-512   # for trivia
# LOAD_CKPT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-5120   # for nq
# LOAD_CKPT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-51200   # for macro
LOAD_CKPT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-128   # for macro


CUDA_VISIBLE_DEVICES=0 python3 -m IDGen.finalize_ckpt  \
    --output_dir $LOAD_CKPT_DIR/final  \
    --model_name_or_path $LOAD_CKPT_DIR  \
    --encoder_type seq2seq \
    --do_train  \
    --save_steps 1000  \
    --eval_steps 200 \
    --logging_steps 100 \
    --target_position 3 \
    --hn_num 1 \
    --add_quantization False \
    --gumbel_softmax True \
    --quantization_pool_size $quan_size \
    --train_path $LOAD_CKPT_DIR/document.3.${MODEL_SIMPLE_NAME}.tokenized.json  \
    --eval_path $LOAD_CKPT_DIR/document.3.${MODEL_SIMPLE_NAME}.tokenized.json  \
    --per_device_train_batch_size 8  \
    --per_device_eval_batch_size 16 \
    --learning_rate $LR  \
    --max_len 512  \
    --decoder_mlm_probability $MASK_RATIO \
    --num_train_epochs 30  \
    --logging_dir $LOG_DIR/tmp  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to wandb \
    --fix_semanticID_generator False \
    --dataloader_num_workers 0
