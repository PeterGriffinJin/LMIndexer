from collections import defaultdict
import os
import torch
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
from copy import deepcopy
from tqdm import tqdm
from argparse import ArgumentParser

from IPython import embed


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--short_data_name', type=str, required=True)
parser.add_argument('--seed', type=int, default=2023)

args = parser.parse_args()


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_json_rows(file_path):
    data = []
    with open(file_path, "r") as f:
        readin = f.readlines()
        for tmp in tqdm(readin):
            data.append(json.loads(tmp))
    return data

def load_json_rows2(file_path):
    data = []
    with open(file_path, "r") as f:
        readin = f.readlines()
        for tmp in tqdm(readin):
            data.append(eval(tmp))
    return data

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def save_dict_list(dict_list, file_name):
    with open(file_name, 'w') as fout:
        for line in dict_list:
            fout.write(json.dumps(line)+'\n')

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

## set seed
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

short_data_name = args.short_data_name
if not os.path.exists(os.path.join(args.output_dir)):
    os.mkdir(args.output_dir)
if not os.path.exists(os.path.join(args.output_dir, 'preprocess')):
    os.mkdir(os.path.join(args.output_dir, 'preprocess'))

if short_data_name == 'beauty':
    full_data_name = 'Beauty'
elif short_data_name == 'toys':
    full_data_name = 'Toys_and_Games'
elif short_data_name == 'sports':
    full_data_name = 'Sports_and_Outdoors'
else:
    raise NotImplementedError


# return (user item timestamp) sort in get_interaction
def Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    data_file = f'{args.data_dir}/review_core_2014/reviews_' + dataset_name + '.json'
    readin = load_json_rows(data_file)

    for inter in tqdm(readin):
        if float(inter['overall']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas

def Amazon_meta(dataset_name, data_maps):
    '''
    asin - ID of the product, e.g. 0000031852
    --"asin": "0000031852",
    title - name of the product
    --"title": "Girls Ballet Tutu Zebra Hot Pink",
    description
    price - price in US dollars (at time of crawl)
    --"price": 3.17,
    imUrl - url of the product image (str)
    --"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
    related - related products (also bought, also viewed, bought together, buy after viewing)
    --"related":{
        "also_bought": ["B00JHONN1S"],
        "also_viewed": ["B002BZX8Z6"],
        "bought_together": ["B002BZX8Z6"]
    },
    salesRank - sales rank information
    --"salesRank": {"Toys & Games": 211836}
    brand - brand name
    --"brand": "Coxlures",
    categories - list of categories the product belongs to
    --"categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
    '''
    datas = {}
    meta_file = f'{args.data_dir}/metadata_2014/meta_' + dataset_name + '.json'
    readin = load_json_rows2(meta_file)

    item_asins = list(data_maps['item2id'].keys())
    for info in tqdm(readin):
        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info
    return datas

def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

def get_interaction(datas):
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq

# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # 已经保证Kcore

# 循环过滤 K-core
def filter_Kcore(user_items, user_core, item_core): # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core: # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items

def id_map(user_items): # user_items dict
    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps


def main(data_name, acronym, data_type='Amazon'):
    assert data_type in {'Amazon'}
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0

    # read in user/item interaction data with time stamps
    datas = Amazon(data_name+'_5', rating_score=rating_score)

    user_items = get_interaction(datas)
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    
    # raw_id user: [item1, item2, item3...]
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    # ID mapping for user/item
    user_items, user_num, item_num, data_maps = id_map(user_items)

    # Statistics on user/item
    user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    # extract metadata
    print('Begin extracting meta infos...')
    meta_infos = Amazon_meta(data_name, data_maps)

    # -------------- Save Data ---------------
    data_file = os.path.join(args.output_dir, 'preprocess', 'sequential_data.txt')
    datamaps_file = os.path.join(args.output_dir, 'preprocess', 'datamaps.json')

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')

    json_str = json.dumps(data_maps)
    with open(datamaps_file, 'w') as out:
        out.write(json_str)


def sample_test_data(data_name, test_num=99, sample_type='random'):
    """
    sample_type:
        random:  sample `test_num` negative items randomly.
        pop: sample `test_num` negative items according to item popularity.
    """

    data_file = f'sequential_data.txt'
    test_file = f'negative_samples.txt'

    item_count = defaultdict(int)
    user_items = defaultdict()

    lines = open('./{}/preprocess/'.format(data_name) + data_file).readlines()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_items[user] = items
        for item in items:
            item_count[item] += 1

    all_item = list(item_count.keys())
    count = list(item_count.values())
    sum_value = np.sum([x for x in count])
    probability = [value / sum_value for value in count]

    user_neg_items = defaultdict()

    print('Begin sampling testing items ...')
    for user, user_seq in tqdm(user_items.items()):
        test_samples = []
        while len(test_samples) < test_num:
            if sample_type == 'random':
                sample_ids = np.random.choice(all_item, test_num, replace=False)
            else: # sample_type == 'pop':
                sample_ids = np.random.choice(all_item, test_num, replace=False, p=probability)
            sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in test_samples]
            test_samples.extend(sample_ids)
        test_samples = test_samples[:test_num]
        user_neg_items[user] = test_samples

    with open(os.path.join(args.output_dir, 'preprocess', test_file), 'w') as out:
        for user, samples in user_neg_items.items():
            out.write(user+' '+' '.join(samples)+'\n')

main(full_data_name, short_data_name, data_type='Amazon')
sample_test_data(short_data_name)

# meta data process
datamaps = load_json(os.path.join(args.output_dir, 'preprocess', "datamaps.json"))
meta_data = load_json_rows2("{}/metadata_2014/meta_{}.json".format(args.data_dir, full_data_name))
exist_meta_data = []
for tmp_meta in tqdm(meta_data):
    if tmp_meta['asin'] not in datamaps['item2id']:
        continue
    save_meta = deepcopy(tmp_meta)
    save_meta['item_id'] = datamaps['item2id'][tmp_meta['asin']]
    exist_meta_data.append(save_meta)

print(len(exist_meta_data))

# save exist_meta_data
save_dict_list(exist_meta_data, os.path.join(args.output_dir, 'preprocess', "meta_all.jsonl"))
