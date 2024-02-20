# This code script is for warming up the decoder.

DOMAIN=beauty
PROCESSED_DIR=~/quic-efs/user/bowenjin/SemanticID/data/$DOMAIN
LOG_DIR=~/quic-efs/user/bowenjin/SemanticID/logs/$DOMAIN
CHECKPOINT_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN

MODEL='t5-base'      #   sentence-transformers/all-mpnet-base-v2

# LR="5e-3"
MASK_RATIO=0

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

match=$(echo $MODEL | grep "/")
if [[ "$match"=="$MODEL" ]]
then
    MODEL_SIMPLE_NAME=$(echo $MODEL | cut -d "/" -f 2)
else
    MODEL_SIMPLE_NAME=$MODEL
fi

echo "start training..."

# with quantization (optimize encoder & codes)
MODEL_TYPE='GB'
quan_size=512

####### train epoch 30 for amazon datasets, cpu worker 16, lr 1e-3, eval_step 200, maxlen 512
####### train epoch 10 for NQ dataset, eval_step 1000, maxlen 512

# for LR in 2e-3 1e-3 5e-4
for LR in 1e-3
do
torchrun --nproc_per_node=8 --master_port 19324 \
    -m IDGen.run  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/warm_decoder/$LR/$MASK_RATIO/quantization-$quan_size  \
    --model_name_or_path "t5-base"  \
    --encoder_type seq2seq \
    --do_train  \
    --save_steps 5000  \
    --eval_steps 200  \
    --logging_steps 100 \
    --add_quantization True \
    --gumbel_softmax True \
    --fix_GB_codebook True \
    --quantization_pool_size $quan_size \
    --train_path $PROCESSED_DIR/document.${MODEL_SIMPLE_NAME}.tokenized.json  \
    --eval_path $PROCESSED_DIR/document.${MODEL_SIMPLE_NAME}.tokenized.json  \
    --dataset_name $DOMAIN \
    --per_device_train_batch_size 16  \
    --per_device_eval_batch_size 16 \
    --learning_rate $LR  \
    --max_len 512  \
    --decoder_mlm_probability $MASK_RATIO \
    --num_train_epochs 30  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/warm_decoder/$LR/$MASK_RATIO/quantization-$quan_size  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to wandb \
    --fix_semanticID_generator True \
    --dataloader_num_workers 16
done
