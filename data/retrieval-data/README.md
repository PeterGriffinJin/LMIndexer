
## prepare data for TriviaQA
### run the cells in Trivia_dataset_Process.ipynb
```
Trivia_dataset_Process.ipynb
```

### fast tokenize
```
cd ../src/embedding
bash fast_tokenize_qa.sh
```

### encode
```
bash encode.sh
```

### generate ids

1. RQ-VAE
```
cd ../rqvae
bash run.sh
```

2. Tree clustering
```
cd ../src/embedding
bash construct_tree.sh
```

### generate final train/val/test files
```
bash retrieval_gen.sh
```
