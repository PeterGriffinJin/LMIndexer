# Pipeline for sequential recommendation

## Raw data preprocess.
```
cd rec-data
bash data_preprocess.sh
```

## Sequential generative recommendation data processing.
### fast tokenize (for rqvae/tree).
```
cd ../src/embedding
bash fast_tokenize.sh
```

### Generate embeddings (for rqvae/tree).
```
bash encode.sh
```

### Generate semantic ids.

1. vanilla PQ/RVQ (not maintained)
```
bash pq.sh
```

2. RQ-VAE
```
cd ../rqvae
bash run.sh
```

3. Tree clustering
```
cd ../src/embedding
bash construct_tree.sh
```

4. ours
```
cd .../SemanticID
```
Run the code their to obtain the semantic ID generator model. Follow the README there.

### Generate final train/val/test files.
```
cd ..
cd ../rec-data
bash sqrec_gen.sh
```
Change **SEMANTIC_MODE** to atomic/rqvae/tree/ours


## Sequential generative recommendation training.

Train the baselines.
```
cd src/
bash run_baseline_rec.sh
```

Train our model.
```
cd src/
bash run_ours_rec.sh
```

## Sequential generative recommendation testing.

```
bash test_rec.sh
```
