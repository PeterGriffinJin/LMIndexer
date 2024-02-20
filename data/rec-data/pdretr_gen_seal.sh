DOMAIN=toys

RAW_RETRIEVAL_DATA_DIR=../raw_data/retrieval/esci-data/shopping_queries_dataset
RAW_ALL_ITEM_FILE=../raw_data/metadata_2014/metadata.json
INTERMEDIATE_DIR=../rec-data
BM25_DIR=../rec-data/$DOMAIN/query_retrieval/bm25
INTERMEDIATE_ITEM_FILE=../rec-data/$DOMAIN/preprocess/meta_all.jsonl

python pdretr_gen_seal.py \
    --raw_retrieval_data_dir $RAW_RETRIEVAL_DATA_DIR \
    --raw_item_file $RAW_ALL_ITEM_FILE \
    --intermediate_dir $INTERMEDIATE_DIR \
    --intermediate_item_file $INTERMEDIATE_ITEM_FILE \
    --bm25_dir $BM25_DIR \
    --domain $DOMAIN \
    --max_rank 30
