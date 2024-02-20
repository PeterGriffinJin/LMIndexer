# This code script is for training encoder/decoder together without codebook components.

DOMAIN=beauty
PROCESSED_DIR=~/quic-efs/user/bowenjin/SemanticID/data/$DOMAIN
LOG_DIR=~/quic-efs/user/bowenjin/SemanticID/logs/$DOMAIN
CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN

MODEL='t5-base'      #   sentence-transformers/all-mpnet-base-v2

LR="1e-2"
PRETRAIN_MASK_RATIO=1
MASK_RATIO=0.5

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
quan_size=256

# # learn all the components from scratch
# for LR in 1e-3
# do
# torchrun --nproc_per_node=8 --master_port 19330 \
#     -m IDGen.run  \
#     --output_dir $CHECKPOINT_DIR/debug  \
#     --model_name_or_path $MODEL  \
#     --encoder_type seq2seq \
#     --do_train  \
#     --save_steps 1000  \
#     --eval_steps 200 \
#     --logging_steps 100 \
#     --add_quantization False \
#     --gumbel_softmax True \
#     --quantization_pool_size $quan_size \
#     --train_path $PROCESSED_DIR/document.${MODEL_SIMPLE_NAME}.tokenized.json  \
#     --eval_path $PROCESSED_DIR/document.${MODEL_SIMPLE_NAME}.tokenized.json  \
#     --per_device_train_batch_size 16  \
#     --per_device_eval_batch_size 16 \
#     --learning_rate $LR  \
#     --max_len 512  \
#     --decoder_mlm_probability $MASK_RATIO \
#     --num_train_epochs 30  \
#     --logging_dir $LOG_DIR/debug  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --report_to wandb \
#     --fix_semanticID_generator False \
#     --dataloader_num_workers 16
# done


# # load the pretrained decoder parameters
# LOAD_CKPT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/warm_decoder/1e-3/0/quantization-512
# LOAD_CKPT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/warm_decoder/1e-3/0/quantization-51200
LOAD_CKPT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/warm_decoder/1e-3/0/quantization-$quan_size

# without encoder contrastive
# for LR in 1e-3
# do
# torchrun --nproc_per_node=8 --master_port 19331 \
#     -m IDGen.run  \
#     --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/nocode/nocontrastive/$LR  \
#     --model_name_or_path $LOAD_CKPT_DIR  \
#     --encoder_type seq2seq \
#     --do_train  \
#     --save_steps 1000  \
#     --eval_steps 200 \
#     --logging_steps 100 \
#     --add_quantization False \
#     --gumbel_softmax True \
#     --quantization_pool_size $quan_size \
#     --train_path $PROCESSED_DIR/document.${MODEL_SIMPLE_NAME}.tokenized.json  \
#     --eval_path $PROCESSED_DIR/document.${MODEL_SIMPLE_NAME}.tokenized.json  \
#     --dataset_name $DOMAIN \
#     --per_device_train_batch_size 16  \
#     --per_device_eval_batch_size 16 \
#     --learning_rate $LR  \
#     --max_len 512  \
#     --decoder_mlm_probability $MASK_RATIO \
#     --num_train_epochs 30  \
#     --logging_dir $LOG_DIR/$MODEL_TYPE/nocode/nocontrastive/$LR  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --report_to wandb \
#     --fix_semanticID_generator False \
#     --dataloader_num_workers 16
# done


####### train epoch 30 for amazon datasets, cpu worker 16, lr 1e-3, eval_step 200
####### train epoch 10 for NQ dataset, cpu worker 16, lr 1e-3, eval_step 1000


# with encoder contrastive
for LR in 1e-3
do
torchrun --nproc_per_node=8 --master_port 19331 \
    -m IDGen.run  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/position1/nocode/contrastive/$LR  \
    --model_name_or_path $LOAD_CKPT_DIR  \
    --encoder_type seq2seq \
    --do_train  \
    --save_steps 5000  \
    --eval_steps 200 \
    --logging_steps 100 \
    --add_quantization False \
    --gumbel_softmax True \
    --contrastive_obj True \
    --quantization_pool_size $quan_size \
    --train_path $PROCESSED_DIR/document.${MODEL_SIMPLE_NAME}.tokenized.json  \
    --eval_path $PROCESSED_DIR/document.${MODEL_SIMPLE_NAME}.tokenized.200000.json  \
    --dataset_name $DOMAIN \
    --per_device_train_batch_size 16  \
    --per_device_eval_batch_size 16 \
    --learning_rate $LR  \
    --max_len 128  \
    --decoder_mlm_probability $MASK_RATIO \
    --num_train_epochs 30  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/position1/nocode/contrastive/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to wandb \
    --fix_semanticID_generator False \
    --dataloader_num_workers 16
done
