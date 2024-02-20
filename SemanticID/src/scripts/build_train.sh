DOMAIN=beauty

PROCESSED_DIR=~/quic-efs/user/bowenjin/SemanticID/data/$DOMAIN
TOKENIZER=t5-base

# max length is 1024 for amazon
# echo "build train for documents..."
# python build_train.py \
#         --data_dir $PROCESSED_DIR \
#         --tokenizer $TOKENIZER \
#         --max_length 128

echo "build train for documents..."
python build_train_zeroshot.py \
        --data_dir $PROCESSED_DIR \
        --tokenizer $TOKENIZER \
        --max_length 1024

# head -n 1000 $PROCESSED_DIR/document.tokenized.json > val.debug.tokenized.json
