DOMAIN=beauty

# TRAIN_FILE=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/document.3.t5-base.tokenized.json   ## amazon
# SAVE_FILE=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/document.3.final.json  ## amazon

# TRAIN_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-5120/document.3.t5-base.tokenized.json   ## nq
# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-5120/document.3.final.json   ## nq

# TRAIN_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-51200/document.3.t5-base.tokenized.json   ## macro
# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-51200/document.3.final.json   ## macro

# TRAIN_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/document.3.t5-base.tokenized.zeroshot.json   ## macro
# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/document.3.final.zeroshot.json   ## macro

# TRAIN_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/3/document.3.t5-base.tokenized.json
# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/3/document.3.final.json

TRAIN_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-128/document.3.t5-base.tokenized.json
SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-128/document.3.final.json

python extract_semanticid_from_train.py \
    --train_file $TRAIN_FILE \
    --save_file $SAVE_FILE

# cp $SAVE_FILE ~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/preprocess/ours-code.json   # amazon
# cp $SAVE_FILE ~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/Trivia_dataset/ours-code.json   # trivia
# cp $SAVE_FILE ~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/NQ_dataset/ours-code.json   # nq
# cp $SAVE_FILE ~/quic-efs/user/bowenjin/seq2seq/retrieval-data/NCI_Data_process/macro_dataset/ours-code.json   # nq
# cp $SAVE_FILE ~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/preprocess/ours-code-zeroshot.json   # nq
# cp $SAVE_FILE ~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/preprocess/ours-recon3-code.json
cp $SAVE_FILE ~/quic-efs/user/bowenjin/seq2seq/rec-data/$DOMAIN/preprocess/ours-dim128-code.json
