
# this will generate ids and save to out_dir/pid_labelid.memmap, use pd.read_csv to read the result

BASE_MODEL='bert-base-uncased'  # sentence-transformers/all-mpnet-base-v2    bert-base-uncased

EMBEDDING_DIR=/home/ec2-user/quic-efs/user/bowenjin/seq2seq/rec-data/beauty/preprocess
CODE_SAVE_DIR=/home/ec2-user/quic-efs/user/bowenjin/seq2seq/rec-data/beauty/preprocess

PARENT_CHILDREN_NUM=50
MAX_DOC_LEAF=100 # must be larger or equal to PARENT_CHILDREN_NUM

python construct_tree.py \
    --embedding_dir $EMBEDDING_DIR \
    --code_save_dir $CODE_SAVE_DIR \
    --base_model $BASE_MODEL \
    --balance_factor $PARENT_CHILDREN_NUM \
    --leaf_factor $MAX_DOC_LEAF
