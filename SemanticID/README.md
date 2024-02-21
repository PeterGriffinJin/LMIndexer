# LMIndexer: Learning self-supervised semantic indexer with generative language model

## Overview
**LMIndexer** is a self-supervised framework learned to tokenize documents into semantic IDs.

<p align="center">
  <img src="fig/main.png" width="600px"/>
</p>

## Data Preparation

```
cd src/
cd scripts/
bash prepare_data.sh
bash build_train.sh
```

## Run the model (position 1 ID)

### Warm up the reconstructor (fix encoder & codebook)
```
bash warm_up_GB.sh
```

### Train the model without codebook
```
bash nocode.sh
```

### Obtain document embeddings and codebook initializations from the model trained above
```
bash generate_embed.sh
cd scripts
bash embed_clustering.sh
```

### Train the whole model (including encoder, codebook and reconstructor)
```
bash contra_recon_encdec.sh
```

### generate code
```
bash generate_code.sh
```

### Generate new train file
```
cd scripts
bash gen_hard_neg.sh
```

## Run the model (position 2+ ID)

### Train the model without codebook
```
bash nocode_seq.sh
```


### Obtain document embeddings and codebook initializations from the model trained above
```
bash generate_embed_seq.sh
cd scripts
bash embed_clustering.sh
```

### Train the whole model (including encoder, codebook and reconstructor)
```
bash contra_recon_encdec_seq.sh
```

### generate code
```
bash generate_code_seq.sh
```

### Generate new train file
```
cd scripts
bash gen_hard_neg_seq.sh
```

## Finalizing ...

### Finalize model
```
bash finalize_ckpt.sh
```

### Generate final code
```
cd scripts
bash extract_semanticid_from_train.sh
```
