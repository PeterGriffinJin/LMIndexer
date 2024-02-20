domain=beauty

mkdir $domain/

python data_preprocess.py \
    --data_dir ../raw_data \
    --output_dir ./$domain \
    --short_data_name $domain
