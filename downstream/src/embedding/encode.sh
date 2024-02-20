BASE_MODEL='bert-base-uncased' # bert-base-uncased

# DOMAIN=toys
# DATA_DIR=~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/preprocess
# OUTPUT_DIR=~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/preprocess

DATA=NQ
DATA_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/${DATA}_dataset
OUTPUT_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/${DATA}_dataset

python encode.py \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --tokenizer $BASE_MODEL \
        --model_name $BASE_MODEL \
        --batch_size 512 \
        --max_length 256 \
        --dataloader_num_workers 0
