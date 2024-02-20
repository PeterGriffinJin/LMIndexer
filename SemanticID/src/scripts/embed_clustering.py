import numpy as np
import time
import os
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import pickle as pkl
from dataclasses import dataclass, field
import torch
import json
from argparse import ArgumentParser

from transformers import HfArgumentParser
import faiss

from IPython import embed

parser = ArgumentParser()
parser.add_argument('--embed_file', type=str, required=True)
parser.add_argument('--save_file', type=str, required=True)
# parser.add_argument('--plm', type=str, required=True)
parser.add_argument('--n_cluster', type=int, required=True)
parser.add_argument('--kmeans_pkg', type=str, default='sklearn')

args = parser.parse_args()

# plm_name = args.plm.split('/')[-1]

# read embedding
# embeddings = torch.load(os.path.join(args.data_dir, f'{plm_name}-embed.pt'), map_location=torch.device('cpu'))
embeddings = torch.load(args.embed_file, map_location=torch.device('cpu'))

if args.kmeans_pkg == "sklearn":
    kmeans = KMeans(n_clusters=args.n_cluster, max_iter=3000, n_init=10).fit(embeddings)
    np.save(open(args.save_file, 'wb'), kmeans.cluster_centers_)
elif args.kmeans_pkg == "faiss":
    kmeans = faiss.Kmeans(embeddings.shape[1], k=args.n_cluster, niter=3000, nredo=10, gpu=True)
    kmeans.train(embeddings)
    centroids = kmeans.centroids
    np.save(open(args.save_file, 'wb'), centroids)
else:
    raise ValueError('Wrong kmeans package name!')

# save center embeddings
# np.save(open(os.path.join(args.output_dir, f'kmeans_center.npy'), 'wb'), kmeans.cluster_centers_)
