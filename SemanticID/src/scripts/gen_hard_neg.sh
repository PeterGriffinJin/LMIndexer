DOMAIN=beauty

# DATA_FILE=~/quic-efs/user/bowenjin/SemanticID/data/$DOMAIN/document.t5-base.tokenized.zeroshot.json
DATA_FILE=~/quic-efs/user/bowenjin/SemanticID/data/$DOMAIN/document.t5-base.tokenized.json
# CODE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/position1/GB/load_soft_encdec_kmeans_code/1e-3/0.5/quantization-512/semanticid.txt
# CODE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/5e-3/0.5/quantization-5120/semanticid.txt
# CODE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-51200/semanticid.txt
# CODE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/2e-3/0.5/quantization-512/semanticid_zeroshot.txt
# CODE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-512/3/semanticid.txt
CODE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-128/semanticid.txt

# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/position1/GB/load_soft_encdec_kmeans_code/1e-3/0.5/quantization-512/document.2.t5-base.tokenized.json
# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/5e-3/0.5/quantization-5120/document.2.t5-base.tokenized.json
# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-51200/document.2.t5-base.tokenized.json
# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/2e-3/0.5/quantization-512/document.2.t5-base.tokenized.zeroshot.json
# SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-512/3/document.2.t5-base.tokenized.json
SAVE_FILE=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/load_soft_encdec_kmeans_code/contrastive/1e-3/0.5/quantization-128/document.2.t5-base.tokenized.json


python3 gen_hard_neg.py \
    --data_file $DATA_FILE \
    --code_file $CODE_FILE \
    --save_file $SAVE_FILE
