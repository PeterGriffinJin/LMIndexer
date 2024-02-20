# TREC process for macro

version=2020

FILE_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/macro_dataset/TREC_DL_$version
QREL_FILE=$FILE_DIR/qrel.json
QUERY_BINARY_FILE=$FILE_DIR/qrel_binary.json
QUERY_FILE=$FILE_DIR/queries_$version/raw.tsv

SEMANTICID_DATA_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/macro_dataset
CORPUS_FILE=$SEMANTICID_DATA_DIR/corpus.tsv

SAVE_DIR=$FILE_DIR

SEMANTIC_MODE=ours
BASE_MODEL='bert-base-uncased'  #  sentence-transformers/all-mpnet-base-v2

python trec_process_semanticid.py \
    --qrel_file $QREL_FILE \
    --query_file $QUERY_FILE \
    --qrel_binary_file $QUERY_BINARY_FILE \
    --corpus_file $CORPUS_FILE \
    --save_dir $SAVE_DIR \
    --semanticid_mode $SEMANTIC_MODE \
    --semanticid_data_dir $SEMANTICID_DATA_DIR \
    --base $BASE_MODEL
