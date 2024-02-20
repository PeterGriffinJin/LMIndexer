# TREC process for macro

version=2020

FILE_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/macro_dataset/TREC_DL_$version
QREL_FILE=$FILE_DIR/qrel.json
QUERY_BINARY_FILE=$FILE_DIR/qrel_binary.json
QUERY_FILE=$FILE_DIR/queries_$version/raw.tsv
CORPUS_FILE=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/macro_dataset/corpus.tsv
DOCID2IDX_FILE=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/processed/macro/dpr/docid2idx.pkl
SAVE_DIR=$FILE_DIR

python trec_process_bm25_dpr.py \
    --qrel_file $QREL_FILE \
    --query_file $QUERY_FILE \
    --qrel_binary_file $QUERY_BINARY_FILE \
    --corpus_file $CORPUS_FILE \
    --docid2idx_file $DOCID2IDX_FILE \
    --save_dir $SAVE_DIR
