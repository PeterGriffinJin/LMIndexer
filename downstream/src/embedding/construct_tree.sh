
# this will generate ids and save to out_dir/pid_labelid.memmap, use pd.read_csv to read the result

# BASE_MODEL='bert-base-uncased'  # bert-base-uncased
BASE_MODEL='self-contrastive'

DOMAIN=beauty
EMBEDDING_DIR=~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/preprocess
CODE_SAVE_DIR=~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/preprocess

# DOMAIN=macro
# EMBEDDING_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/${DOMAIN}_dataset
# CODE_SAVE_DIR=~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/${DOMAIN}_dataset

# Amazon
# PARENT_CHILDREN_NUM=50
# MAX_DOC_LEAF=512 # must be larger or equal to PARENT_CHILDREN_NUM

# NQ
# PARENT_CHILDREN_NUM=50 # try to decrease this number to 50 next time
# MAX_DOC_LEAF=1280 # must be larger or equal to PARENT_CHILDREN_NUM

# macro
# PARENT_CHILDREN_NUM=50 # try to decrease this number to 50 next time
# MAX_DOC_LEAF=12800 # must be larger or equal to PARENT_CHILDREN_NUM

# rebuttal self-contrastive
PARENT_CHILDREN_NUM=50
MAX_DOC_LEAF=512 # must be larger or equal to PARENT_CHILDREN_NUM


python construct_tree.py \
    --embedding_dir $EMBEDDING_DIR \
    --code_save_dir $CODE_SAVE_DIR \
    --base_model $BASE_MODEL \
    --balance_factor $PARENT_CHILDREN_NUM \
    --leaf_factor $MAX_DOC_LEAF \
    --kmeans_pkg sklearn
