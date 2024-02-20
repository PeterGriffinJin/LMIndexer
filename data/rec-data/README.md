
## prepare data for sequential recommendation

### Data Preprocess
#### Preprocess
```
bash data_preprocess.sh
```

#### Fast tokenize
```
cd ../src/embedding
bash fast_tokenize.sh
```

### Encode
```
bash encode.sh
```

### Generate final train/val/test files
Sequential recommendation
```
cd ..
cd ../rec-data
bash sqrec_gen.sh
```

Product retrieval
```
bash pdretr_gen.sh
```
