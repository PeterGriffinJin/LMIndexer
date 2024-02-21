DOMAIN=beauty

DATA_DIR=./$DOMAIN
SEMANTIC_MODE=ours # tree rqvae
BASE_MODEL='bert-base-uncased'

python sqrec_gen.py \
    --data_dir $DATA_DIR \
    --semantic_id_mode $SEMANTIC_MODE \
    --base $BASE_MODEL \
    --train_iter 5
