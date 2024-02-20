# This code script is for training encoder/codebook/decoder together without warming up the codebook and decoder.

DOMAIN=beauty
LOG_DIR=~/quic-efs/user/bowenjin/SemanticID/logs/$DOMAIN
CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN

MODEL='t5-base'      #   sentence-transformers/all-mpnet-base-v2

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

# load kmeans clustered codebook embeddings
# LOAD_CKPT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/nocode/contrastive/1e-3
LOAD_CKPT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/nocode/contrastive/1e-3/quantization-$quan_size

CODEBOOK_FILE=$LOAD_CKPT_DIR/kmeans_center.npy

# LOAD_CORPORA_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/position1/GB/load_soft_encdec_kmeans_code/1e-3/0.5/quantization-512
# LOAD_CORPORA_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/position1/GB/load_soft_encdec_kmeans_code/contrastive/2e-3/0.5/quantization-512
# LOAD_CORPORA_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/5e-3/0.5/quantization-5120
# LOAD_CORPORA_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-51200
LOAD_CORPORA_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-$quan_size

for LR in 1e-3
do
torchrun --nproc_per_node=8 --master_port 19331 \
    -m IDGen.run_neg  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/position2/load_soft_encdec_kmeans_code/contrastive/$LR/$MASK_RATIO/quantization-$quan_size  \
    --model_name_or_path $LOAD_CKPT_DIR  \
    --encoder_type seq2seq \
    --do_train  \
    --save_steps 10000  \
    --eval_steps 800 \
    --logging_steps 100 \
    --target_position 2 \
    --hn_num 4 \
    --add_quantization True \
    --gumbel_softmax True \
    --contrastive_obj True \
    --quantization_pool_size $quan_size \
    --load_codebook True \
    --load_codebook_dir $CODEBOOK_FILE \
    --train_path $LOAD_CORPORA_DIR/document.2.${MODEL_SIMPLE_NAME}.tokenized.json  \
    --eval_path $LOAD_CORPORA_DIR/document.2.${MODEL_SIMPLE_NAME}.tokenized.json  \
    --dataset_name $DOMAIN \
    --per_device_train_batch_size 4  \
    --per_device_eval_batch_size 16 \
    --learning_rate $LR  \
    --max_len 128  \
    --decoder_mlm_probability $MASK_RATIO \
    --num_train_epochs 30  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/position2/load_soft_encdec_kmeans_code/contrastive/$LR/$MASK_RATIO/quantization-$quan_size  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to wandb \
    --fix_semanticID_generator False \
    --dataloader_num_workers 32
done


# for LR in 1e-3
# do
# torchrun --nproc_per_node=8 --master_port 19331 \
#     -m IDGen.run_neg  \
#     --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/position2/soft_and_rand_code/contrastive/$LR/$MASK_RATIO/quantization-$quan_size  \
#     --model_name_or_path $LOAD_CKPT_DIR  \
#     --encoder_type seq2seq \
#     --do_train  \
#     --save_steps 1000  \
#     --eval_steps 800 \
#     --logging_steps 100 \
#     --target_position 2 \
#     --hn_num 4 \
#     --add_quantization True \
#     --gumbel_softmax True \
#     --contrastive_obj True \
#     --quantization_pool_size $quan_size \
#     --load_codebook False \
#     --load_codebook_dir $CODEBOOK_FILE \
#     --train_path $LOAD_CORPORA_DIR/document.2.${MODEL_SIMPLE_NAME}.tokenized.json  \
#     --eval_path $LOAD_CORPORA_DIR/document.2.${MODEL_SIMPLE_NAME}.tokenized.json  \
#     --dataset_name $DOMAIN \
#     --per_device_train_batch_size 4  \
#     --per_device_eval_batch_size 16 \
#     --learning_rate $LR  \
#     --max_len 512  \
#     --decoder_mlm_probability $MASK_RATIO \
#     --num_train_epochs 30  \
#     --logging_dir $LOG_DIR/$MODEL_TYPE/position2/soft_and_rand_code/contrastive/$LR/$MASK_RATIO/quantization-$quan_size  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --report_to wandb \
#     --fix_semanticID_generator False \
#     --dataloader_num_workers 32
# done


# LOAD_CKPT_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/2e-3/0.5/quantization-512
# for LR in 1e-3
# do
# CUDA_VISIBLE_DEVICES=0 python -m IDGen.run_neg  \
#     --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/position2/direct_rand_code/contrastive/$LR/$MASK_RATIO/quantization-$quan_size  \
#     --model_name_or_path $LOAD_CKPT_DIR  \
#     --encoder_type seq2seq \
#     --do_train  \
#     --save_steps 1000  \
#     --eval_steps 800 \
#     --logging_steps 100 \
#     --target_position 2 \
#     --hn_num 4 \
#     --add_quantization True \
#     --gumbel_softmax True \
#     --contrastive_obj True \
#     --quantization_pool_size $quan_size \
#     --load_codebook False \
#     --load_codebook_dir $CODEBOOK_FILE \
#     --train_path $LOAD_CORPORA_DIR/document.2.${MODEL_SIMPLE_NAME}.tokenized.json  \
#     --eval_path $LOAD_CORPORA_DIR/document.2.${MODEL_SIMPLE_NAME}.tokenized.json  \
#     --dataset_name $DOMAIN \
#     --per_device_train_batch_size 4  \
#     --per_device_eval_batch_size 16 \
#     --learning_rate $LR  \
#     --max_len 512  \
#     --decoder_mlm_probability $MASK_RATIO \
#     --num_train_epochs 30  \
#     --logging_dir $LOG_DIR/$MODEL_TYPE/position2/direct_rand_code/contrastive/$LR/$MASK_RATIO/quantization-$quan_size  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --report_to wandb \
#     --fix_semanticID_generator False \
#     --dataloader_num_workers 32
# done

