import os
import json
from tqdm import tqdm
from collections import defaultdict, Counter

from argparse import ArgumentParser

from IPython import embed

parser = ArgumentParser()
parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--save_file', type=str, required=True)
args = parser.parse_args()


# read train file & save to save file
# semanticid_dict = defaultdict(list)

# with open(args.train_file) as f:
#     with open(args.save_file, 'w') as fout:
#         readin = f.readlines()
#         for idd, line in enumerate(tqdm(readin)):
#             tmp = json.loads(line)
#             curr_len = len(tmp['prefix_ids'].split('><'))
#             fout.write(json.dumps({
#                 'token_ids': tmp['token_ids'],
#                 'semantic_id': tmp['prefix_ids'] + f"<id_{curr_len}_{len(semanticid_dict[tmp['prefix_ids']])}>"
#             }) + '\n')
#             semanticid_dict[tmp['prefix_ids']].append(idd)

save_dict = {}
with open(args.train_file) as f:
    readin = f.readlines()
    for idd, line in enumerate(tqdm(readin)):
        tmp = json.loads(line)
        prefix_ids = tmp['prefix_ids'].split('><')
        prefix_ids = [iidds.split("_")[-1] for iidds in prefix_ids]
        prefix_ids[-1] = prefix_ids[-1][:-1]
        prefix_ids = ','.join(prefix_ids)
        save_dict[int(idd+1)] = prefix_ids

with open(args.save_file, 'w') as fout:
    json.dump(save_dict, fout, indent = 4)
