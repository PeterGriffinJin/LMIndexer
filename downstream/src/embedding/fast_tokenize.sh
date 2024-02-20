TOKENIZER='bert-base-uncased'
DOMAIN=toys

INPUT_DIR=downstream/rec-data/$DOMAIN/preprocess
OUTPUT_DIR=downstream/rec-data/$DOMAIN/preprocess

python fast_tokenize.py \
        --input_dir $INPUT_DIR \
        --output $OUTPUT_DIR \
        --tokenizer $TOKENIZER
