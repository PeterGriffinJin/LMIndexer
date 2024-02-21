# Data Processing

## Download the Raw Data
Raw data can be downloaded from [Amazon-Recommendation](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html), [Amazon-Retrieval](https://github.com/amazon-science/esci-data), [NQ](https://drive.google.com/drive/folders/1AcGozhgI679j9ybxL7iCi2iMHipIlHnY?usp=drive_link) and [MS-MACRO](https://drive.google.com/drive/folders/1WQTp7caUyQZXWwoVu2_Tj5NJ56pPsRVj?usp=drive_link).

Put your raw data under ```raw-data/```

## Raw data preprocess.
### Product Recommendation
1. Preprocess the Amazon data. Remember to change the ```domain```. This step is shared for recommendation and product retrieval.
```
cd rec-data/
bash data_preprocess.sh
```

2. Generate recommendation data.
```
bash sqrec_gen.sh
```

### Product Retrieval
1. Generate recommendation data.
```
bash pdretr_gen.sh
```

### Document Retrieval
1. Generate document retrieval data.
```
cd ..
cd retrieval-data/
bash retrieval_gen.sh
```

