
DATA_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/macro_dataset
SEMANTIC_MODE=ours
SAVE_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/macro_dataset/TREC_DL_ALL

# python trec_merge_dpr.py \
#     --data_dir $DATA_DIR \
#     --mode $SEMANTIC_MODE \
#     --save_dir $SAVE_DIR


python trec_merge_semanticid.py \
    --data_dir $DATA_DIR \
    --mode $SEMANTIC_MODE \
    --save_dir $SAVE_DIR
