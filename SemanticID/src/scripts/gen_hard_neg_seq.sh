DOMAIN=beauty

# DATA_FILE=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/2e-3/0.5/quantization-512/document.2.t5-base.tokenized.json
# CODE_FILE=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/semanticid.txt
# SAVE_FILE=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/document.3.t5-base.tokenized.json

# DATA_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/5e-3/0.5/quantization-5120/document.2.t5-base.tokenized.json
# CODE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-5120/semanticid.txt
# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-5120/document.3.t5-base.tokenized.json

# DATA_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-51200/document.2.t5-base.tokenized.json
# CODE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-51200/semanticid.txt
# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-51200/document.3.t5-base.tokenized.json

# DATA_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/2e-3/0.5/quantization-512/document.2.t5-base.tokenized.zeroshot.json
# CODE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/semanticid_zeroshot.txt
# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-512/document.3.t5-base.tokenized.zeroshot.json

DATA_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-128/document.2.t5-base.tokenized.json
CODE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-128/semanticid.txt
SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/load_soft_encdec_kmeans_code/contrastive/1e-3/0.7/quantization-128/document.3.t5-base.tokenized.json


python3 gen_hard_neg_seq.py \
    --data_file $DATA_FILE \
    --code_file $CODE_FILE \
    --save_file $SAVE_FILE
