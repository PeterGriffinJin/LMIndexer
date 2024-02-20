
DOMAIN=beauty
# FILE_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/nocode/nocontrastive/1e-3
# FILE_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/nocode/contrastive/1e-3
# FILE_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/nocode/contrastive/5e-3
# FILE_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position1/nocode/contrastive/1e-3/quantization-128

# FILE_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/pos2/nocode/nocontrastive/1e-3
# FILE_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/pos2/nocode/contrastive/1e-3
# FILE_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/nocode/contrastive/1e-3
# FILE_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/nocode/nocontrastive/1e-3
# FILE_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/nocode/contrastive/1e-3/3
FILE_DIR=~/quic-efs/user/bowenjin/SemanticID/ckpt/$DOMAIN/GB/position2/nocode/contrastive/1e-3/quantization-128

# EMBED_DIR=/home/ec2-user/quic-efs/user/bowenjin/SemanticID/data/$DOMAIN
EMBED_FILE=$FILE_DIR/embed.pt
SAVE_FILE=$FILE_DIR/kmeans_center.npy
KMEANS_PKG=faiss

# PLM=sentence-transformers/all-mpnet-base-v2
N_CLUSTER=128

python3 embed_clustering.py \
    --embed_file $EMBED_FILE \
    --save_file $SAVE_FILE \
    --n_cluster $N_CLUSTER \
    --kmeans_pkg $KMEANS_PKG
