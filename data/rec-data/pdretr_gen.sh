DOMAIN=beauty

RAW_RETRIEVAL_DATA_DIR=../raw_data/retrieval/esci-data/shopping_queries_dataset
RAW_ALL_ITEM_FILE=../raw_data/metadata_2014/metadata.json
INTERMEDIATE_DIR=../rec-data

SEMANTIC_MODE=ours
BASE_MODEL='bert-base-uncased'  #  bert-base-uncased

python pdretr_gen.py \
    --raw_retrieval_data_dir $RAW_RETRIEVAL_DATA_DIR \
    --raw_item_file $RAW_ALL_ITEM_FILE \
    --intermediate_dir $INTERMEDIATE_DIR \
    --domain $DOMAIN \
    --base $BASE_MODEL \
    --semantic_id_mode $SEMANTIC_MODE
