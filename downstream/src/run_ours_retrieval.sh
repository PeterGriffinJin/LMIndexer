
DOMAIN=sports
ID_MODE=ours

NUM_CODES=3
CODEBOOK_SIZE=512

DATA_DIR=/home/ec2-user/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/query_retrieval/$ID_MODE

LOG_DIR=/home/ec2-user/quic-efs/user/bowenjin/seq2seq/logs/$DOMAIN
CHECKPOINT_DIR=/home/ec2-user/quic-efs/user/bowenjin/seq2seq/ckpt/$DOMAIN/query_retrieval/$ID_MODE

SEMANTIC_ID_MODEL_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/final

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# history item text as input, predict target item id
for LR in 1e-2 1e-3
do
torchrun --nproc_per_node=8 --master_port 19288 GEU/run_ours.py \
    --model_name_or_path $SEMANTIC_ID_MODEL_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --save_steps 2000  \
    --eval_steps 1000  \
    --logging_steps 100 \
    --max_steps 10000  \
    --learning_rate $LR  \
    --max_source_length 1024 \
    --max_target_length 128 \
    --add_code_as_special_token True \
    --num_codes $NUM_CODES \
    --codebook_size $CODEBOOK_SIZE \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/test.json \
    --all_id_txt $DATA_DIR/ids.txt \
    --num_beams 20 \
    --output_dir $CHECKPOINT_DIR/$base/$LR  \
    --logging_dir $LOG_DIR/$base/$LR  \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --report_to wandb \
    --overwrite_output_dir True \
    --evaluation_strategy steps \
    --dataloader_num_workers 8 \
    --include_inputs_for_metrics True \
    --project_name 'query retrieval '$DOMAIN \
    --task 'retrieval'
done


#############################################


DOMAIN=beauty
ID_MODE=ours

NUM_CODES=3
CODEBOOK_SIZE=512

DATA_DIR=/home/ec2-user/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/query_retrieval/$ID_MODE

LOG_DIR=/home/ec2-user/quic-efs/user/bowenjin/seq2seq/logs/$DOMAIN
CHECKPOINT_DIR=/home/ec2-user/quic-efs/user/bowenjin/seq2seq/ckpt/$DOMAIN/query_retrieval/$ID_MODE

SEMANTIC_ID_MODEL_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/final

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# history item text as input, predict target item id
for LR in 1e-2 1e-3
do
torchrun --nproc_per_node=8 --master_port 19288 GEU/run_ours.py \
    --model_name_or_path $SEMANTIC_ID_MODEL_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --save_steps 2000  \
    --eval_steps 1000  \
    --logging_steps 100 \
    --max_steps 10000  \
    --learning_rate $LR  \
    --max_source_length 1024 \
    --max_target_length 128 \
    --add_code_as_special_token True \
    --num_codes $NUM_CODES \
    --codebook_size $CODEBOOK_SIZE \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $DATA_DIR/test.json \
    --all_id_txt $DATA_DIR/ids.txt \
    --num_beams 20 \
    --output_dir $CHECKPOINT_DIR/$base/$LR  \
    --logging_dir $LOG_DIR/$base/$LR  \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --report_to wandb \
    --overwrite_output_dir True \
    --evaluation_strategy steps \
    --dataloader_num_workers 8 \
    --include_inputs_for_metrics True \
    --project_name 'query retrieval '$DOMAIN \
    --task 'retrieval'
done

