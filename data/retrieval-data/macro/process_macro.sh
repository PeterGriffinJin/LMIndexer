TRAIN_NUM=1000000
EVAL_NUM=7126
SAVE_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/macro_dataset

python process_macro.py \
    --train_num $TRAIN_NUM \
    --eval_num $EVAL_NUM \
    --save_dir $SAVE_DIR
