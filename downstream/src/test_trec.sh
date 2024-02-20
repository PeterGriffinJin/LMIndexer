# This is the code for query retrieval testing.

# version=2020

base=t5-base
LR=5e-3

DOMAIN=macro
ID_MODE=ours # atomic, rqvae, tree, ours # this is only to identify which files to use for train
BASE_EMBED_MODEL=bert-base-uncased

# atomic id
# NUM_CODES=1
# CODEBOOK_SIZE=1000000   # Trivia 73970, NQ 109739, macro 1000000

# rqvae
# NUM_CODES=4
# CODEBOOK_SIZE=1280

# # tree
# NUM_CODES=3
# CODEBOOK_SIZE=12800

# # ours
NUM_CODES=3
CODEBOOK_SIZE=51200

if [ $ID_MODE = "atomic" ] || [ $ID_MODE = "ours" ];
then
   DATA_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/processed/$DOMAIN/$ID_MODE
   if [ $ID_MODE = "atomic" ];
   then
      LOAD_CHECKPOINT=~/quic-efs/user/bowenjin/seq2seq/ckpt/$DOMAIN/$ID_MODE/$BASE_EMBED_MODEL/$base/$LR
   else
      LOAD_CHECKPOINT=~/quic-efs/user/bowenjin/seq2seq/ckpt/$DOMAIN/$ID_MODE/$LR
   fi
else
   DATA_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/processed/$DOMAIN/$ID_MODE/$BASE_EMBED_MODEL
   LOAD_CHECKPOINT=~/quic-efs/user/bowenjin/seq2seq/ckpt/$DOMAIN/$ID_MODE/$BASE_EMBED_MODEL/$base/$LR
fi

LOG_DIR=~/quic-efs/user/bowenjin/seq2seq/logs/$DOMAIN

# TEST_QUERY_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/macro_dataset/TREC_DL_$version/$ID_MODE
TEST_QUERY_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/macro_dataset/TREC_DL_ALL/$ID_MODE


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --master_port 19289 GEU/run_ours.py \
    --model_name_or_path $LOAD_CHECKPOINT \
    --do_predict \
    --max_source_length 32 \
    --max_target_length 32 \
    --add_code_as_special_token True \
    --num_codes $NUM_CODES \
    --codebook_size $CODEBOOK_SIZE \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $TEST_QUERY_DIR/test.query.json \
    --all_id_txt $DATA_DIR/ids.txt \
    --num_beams 20 \
    --report_to none \
    --output_dir $LOAD_CHECKPOINT/result  \
    --logging_dir $LOG_DIR/$base/$LR  \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 64 \
    --overwrite_output_dir \
    --predict_with_generate \
    --overwrite_output_dir True \
    --dataloader_num_workers 8 \
    --include_inputs_for_metrics True \
    --task 'retrieval' \
    --save_trec True \
    --save_trec_file 'test.pred.trec' \
    --eval_topk 10


torchrun --nproc_per_node=8 --master_port 19289 GEU/run_ours.py \
    --model_name_or_path $LOAD_CHECKPOINT \
    --do_predict \
    --max_source_length 32 \
    --max_target_length 32 \
    --add_code_as_special_token True \
    --num_codes $NUM_CODES \
    --codebook_size $CODEBOOK_SIZE \
    --train_file $DATA_DIR/train.json \
    --validation_file $DATA_DIR/val.json \
    --test_file $TEST_QUERY_DIR/test.query.binary.json \
    --all_id_txt $DATA_DIR/ids.txt \
    --num_beams 20 \
    --report_to none \
    --output_dir $LOAD_CHECKPOINT/result  \
    --logging_dir $LOG_DIR/$base/$LR  \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 64 \
    --overwrite_output_dir \
    --predict_with_generate \
    --overwrite_output_dir True \
    --dataloader_num_workers 8 \
    --include_inputs_for_metrics True \
    --task 'retrieval' \
    --save_trec True \
    --save_trec_file 'test.pred.binary.trec' \
    --eval_topk 10

~/quic-efs/user/bowenjin/seq2seq/bm25/trec_eval/trec_eval -m ndcg $TEST_QUERY_DIR/test.truth.trec $LOAD_CHECKPOINT/result/test.pred.trec
~/quic-efs/user/bowenjin/seq2seq/bm25/trec_eval/trec_eval -c -m recall.1 $TEST_QUERY_DIR/test.truth.binary.trec $LOAD_CHECKPOINT/result/test.pred.binary.trec
~/quic-efs/user/bowenjin/seq2seq/bm25/trec_eval/trec_eval -c -m recall.5 $TEST_QUERY_DIR/test.truth.binary.trec $LOAD_CHECKPOINT/result/test.pred.binary.trec
~/quic-efs/user/bowenjin/seq2seq/bm25/trec_eval/trec_eval -c -m recall.10 $TEST_QUERY_DIR/test.truth.binary.trec $LOAD_CHECKPOINT/result/test.pred.binary.trec
~/quic-efs/user/bowenjin/seq2seq/bm25/trec_eval/trec_eval -m recip_rank $TEST_QUERY_DIR/test.truth.binary.trec $LOAD_CHECKPOINT/result/test.pred.binary.trec
~/quic-efs/user/bowenjin/seq2seq/bm25/trec_eval/trec_eval -m map $TEST_QUERY_DIR/test.truth.binary.trec $LOAD_CHECKPOINT/result/test.pred.binary.trec
