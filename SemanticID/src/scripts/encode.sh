BASE_MODEL='t5-base' # 'sentence-transformers/all-mpnet-base-v2'

DOMAIN=beauty
DATA_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/data/$DOMAIN
OUTPUT_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/data/$DOMAIN

python encode.py \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --tokenizer $BASE_MODEL \
        --model_name $BASE_MODEL \
        --batch_size 32 \
        --max_length 256 \
        --dataloader_num_workers 0
