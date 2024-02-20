DOMAIN=macro

PLM=t5-base
# PROCESSED_DIR=~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/query_retrieval/dpr
PROCESSED_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/processed/$DOMAIN/dpr

mkdir $PROCESSED_DIR/$PLM
python build_train_dpr.py \
        --input_dir $PROCESSED_DIR/ \
        --output $PROCESSED_DIR/$PLM \
        --max_length 128 \
        --tokenizer $PLM
