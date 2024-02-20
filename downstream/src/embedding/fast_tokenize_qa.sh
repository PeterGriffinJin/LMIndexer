TOKENIZER='sentence-transformers/all-mpnet-base-v2'   # sentence-transformers/all-mpnet-base-v2, t5-base
DATA=NQ   # NQ, macro

INPUT_DIR=downstream/retrieval-data/NCI_Data_process/${DATA}_dataset
OUTPUT_DIR=downstream/retrieval-data/NCI_Data_process/${DATA}_dataset

python fast_tokenize_qa.py \
        --input_dir $INPUT_DIR \
        --output $OUTPUT_DIR \
        --tokenizer $TOKENIZER
