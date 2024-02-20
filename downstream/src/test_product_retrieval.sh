# This is the code for query retrieval testing.

base=t5-base
LR=1e-3

DOMAIN=beauty
ID_MODE=ours-dim128 # atomic, rqvae, tree, ours # this is only to identify which files to use for train
BASE_EMBED_MODEL=bert-base-uncased

# atomic id
# NUM_CODES=1 
# CODEBOOK_SIZE=12101   # beauty 12101, toy 11924, sports 18357

# rqvae
# NUM_CODES=4
# CODEBOOK_SIZE=256

# tree
# NUM_CODES=3
# CODEBOOK_SIZE=100

# ours
NUM_CODES=3
CODEBOOK_SIZE=128

if [ $ID_MODE = "atomic" ] || [ $ID_MODE = "ours" ] || [ $ID_MODE = "ours-recon2" ] || [ $ID_MODE = "ours-recon3" ] || [ $ID_MODE = "ours-dim256" ] || [ $ID_MODE = "ours-dim128" ];
then
   DATA_DIR=~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/query_retrieval/$ID_MODE
   if [ $ID_MODE = "atomic" ];
   then
      LOAD_CHECKPOINT=~/quic-efs/user/bowenjin/seq2seq/ckpt/$DOMAIN/query_retrieval/$ID_MODE/$BASE_EMBED_MODEL/$base/$LR
   else
      LOAD_CHECKPOINT=~/quic-efs/user/bowenjin/seq2seq/ckpt/$DOMAIN/query_retrieval/$ID_MODE/$LR
   fi
else
   DATA_DIR=~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/query_retrieval/$ID_MODE/$BASE_EMBED_MODEL
   LOAD_CHECKPOINT=~/quic-efs/user/bowenjin/seq2seq/ckpt/$DOMAIN/query_retrieval/$ID_MODE/$BASE_EMBED_MODEL/$base/$LR
fi

LOG_DIR=~/quic-efs/user/bowenjin/seq2seq/logs/$DOMAIN

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# history item text as input, predict target item id

# torchrun --nproc_per_node=8 --master_port 19289 GEU/run_ours.py \
#     --model_name_or_path $LOAD_CHECKPOINT \
#     --do_predict \
#     --max_source_length 1024 \
#     --max_target_length 128 \
#     --add_code_as_special_token True \
#     --num_codes $NUM_CODES \
#     --codebook_size $CODEBOOK_SIZE \
#     --train_file $DATA_DIR/train.json \
#     --validation_file $DATA_DIR/val.json \
#     --test_file $DATA_DIR/test.json \
#     --all_id_txt $DATA_DIR/ids.txt \
#     --num_beams 20 \
#     --report_to none \
#     --output_dir $LOAD_CHECKPOINT/result  \
#     --logging_dir $LOG_DIR/$base/$LR  \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --overwrite_output_dir \
#     --predict_with_generate \
#     --overwrite_output_dir True \
#     --dataloader_num_workers 8 \
#     --include_inputs_for_metrics True \
#     --task 'retrieval' \
#     --eval_topk 5

CUDA_VISIBLE_DEVICES=0 python GEU/run_ours.py \
    --model_name_or_path $LOAD_CHECKPOINT \
    --do_predict \
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
    --report_to none \
    --output_dir $LOAD_CHECKPOINT/result  \
    --logging_dir $LOG_DIR/$base/$LR  \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 256 \
    --overwrite_output_dir \
    --predict_with_generate \
    --overwrite_output_dir True \
    --dataloader_num_workers 8 \
    --include_inputs_for_metrics True \
    --task 'retrieval' \
    --eval_topk 5
