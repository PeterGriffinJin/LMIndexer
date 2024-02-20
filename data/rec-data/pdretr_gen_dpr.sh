DOMAIN=sports

RAW_RETRIEVAL_DATA_DIR=../raw_data/retrieval/esci-data/shopping_queries_dataset
RAW_ALL_ITEM_FILE=~/quic-efs/user/bowenjin/seq2seq/raw_data/metadata_2014/metadata.json
INTERMEDIATE_DIR=~/quic-efs/user/bowenjin/seq2seq/rec-data
BM25_DIR=~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/query_retrieval/bm25

python pdretr_gen_dpr.py \
    --raw_retrieval_data_dir $RAW_RETRIEVAL_DATA_DIR \
    --raw_item_file $RAW_ALL_ITEM_FILE \
    --intermediate_dir $INTERMEDIATE_DIR \
    --bm25_dir $BM25_DIR \
    --domain $DOMAIN \
    --max_rank 30
