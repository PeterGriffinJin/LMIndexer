
PROCESSED_DIR="data_dir/amazon/sports"

echo "build train for pretrain..."
python build_train.py \
        --input_dir $PROCESSED_DIR \
        --output $PROCESSED_DIR
