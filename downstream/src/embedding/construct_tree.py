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

from transformers import HfArgumentParser
import faiss

from IPython import embed

@dataclass
class TreeConstructArguments:
    embedding_dir: str = field(default="")
    code_save_dir: str = field(default="")
    base_model: str = field(default="")
    balance_factor: int = field(default=10)  # size of each codebook / number of children nodes for each parent nodes
    leaf_factor: int = field(default=30) # number of doc in each leaf nodes to stop clustering
    sampled_doc_num: int = field(default=2_000_000)
    kmeans_pkg: str = field(default="faiss")

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def _node_list(root):
        def node_val(node):
            if(node.isleaf == False):
                return node.val
            else:
                return node.val
            
        node_queue = [root]
        arr_arr_node = []
        arr_arr_node.append([node_val(node_queue[0])])
        while node_queue:
            tmp = []
            tmp_val = []
            for node in node_queue:
                for child in node.children: 
                    tmp.append(child)
                    tmp_val.append(node_val(child))
            if len(tmp_val) > 0:
                arr_arr_node.append(tmp_val)
            node_queue = tmp
        return arr_arr_node


class TreeNode(object):
    """define the tree node structure."""
    def __init__(self, x ,item_embedding = None, layer = None):
        self.val = x   
        self.embedding = item_embedding  
        self.parent = None
        self.children = []
        self.isleaf = False
        self.pids = []
        self.layer = layer
    
    def getval(self):
        return self.val

    def getchildren(self):
        return self.children

    def add(self, node):
            ##if full
        if len(self.children) > 2560:
            raise ValueError(f"length of children shouldn't larger than 50, but now is {len(self.children)}")
        else:
            self.children.append(node)
 

class TreeInitialize(object):
    """"Build the random binary tree."""
    def __init__(self, pid_embeddings, pids, balance_factor, leaf_factor, sampled_doc_num, kmeans_pkg):  
        self.embeddings = pid_embeddings
        self.pids = pids
        self.root = None
        self.balance_factor = balance_factor
        self.leaf_factor = leaf_factor
        self.leaf_dict = {}
        self.node_dict = {}
        self.node_size = 0
        self.sampled_doc_num = sampled_doc_num
        self.kmeans_pkg = kmeans_pkg

    def _k_means_clustering(self, pid_embeddings): 
        if len(pid_embeddings)>self.sampled_doc_num:
            idxs = np.arange(pid_embeddings.shape[0])
            np.random.shuffle(idxs)
            idxs = idxs[0:self.sampled_doc_num]
            train_embeddings = pid_embeddings[idxs] 
        else:
            train_embeddings = pid_embeddings

        if self.kmeans_pkg == "sklearn":
            kmeans = KMeans(n_clusters=self.balance_factor, max_iter=3000, n_init=10).fit(train_embeddings)
            return kmeans
        elif self.kmeans_pkg == "faiss":
            kmeans = faiss.Kmeans(train_embeddings.shape[1], k=self.balance_factor, niter=3000, nredo=10, gpu=True)
            kmeans.train(train_embeddings)
            labels = kmeans.index.search(pid_embeddings, 1)[1].reshape(-1)
            centroids = kmeans.centroids
            print("shape of train_embeddings: ", train_embeddings.shape, self.sampled_doc_num)
            return kmeans, (labels, centroids)
        else:
            raise ValueError(f"{self.kmeans_pkg} is not valid kmeans package.")


    def _build_ten_tree(self, root, pid_embeddings, pids, layer):
        if len(pids) < self.leaf_factor:
            root.isleaf = True
            root.pids = pids
            self.leaf_dict[root.val] = root
            return root

        if self.kmeans_pkg == "sklearn":
            kmeans = self._k_means_clustering(pid_embeddings)
            clusters_embeddings = kmeans.cluster_centers_
            labels = kmeans.labels_
        elif self.kmeans_pkg == "faiss":
            _, (labels, clusters_embeddings) = self._k_means_clustering(pid_embeddings)
            print(labels.shape, clusters_embeddings.shape)
        else:
            raise ValueError(f"{self.kmeans_pkg} is not valid kmeans package.")
        
        for i in range(self.balance_factor):
            val = root.val + "_" + str(i)
            node = TreeNode(x = val, item_embedding=clusters_embeddings[i],layer=layer+1)
            node.parent = root
            index = np.where(labels == i)[0]
            pid_embedding = pid_embeddings[index]
            pid = pids[index]
            node = self._build_ten_tree(node, pid_embedding, pid, layer+1)
            root.add(node)
        return root

    def clustering_tree(self):  
        root = TreeNode('0')
        self.root = self._build_ten_tree(root, self.embeddings, self.pids, layer = 0)
        return self.root

    
if __name__ == '__main__':
    parser = HfArgumentParser((TreeConstructArguments))
    args = parser.parse_args_into_dataclasses()[0]
    
    ## build tree
    output_path = args.code_save_dir
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    tree_path = f"{output_path}/tree.pkl"
    dict_label = {}

    base_model = args.base_model.split('/')[-1]
    # pid_embeddings_all = torch.load(os.path.join(args.embedding_dir, f'{base_model}-embed.pt'), map_location='cpu').numpy()
    pid_embeddings_all = torch.load(os.path.join(args.embedding_dir, f'{base_model}-embed.pt'), map_location='cpu')
    print("size of pid_embeddings_all = {}".format(len(pid_embeddings_all)))

    pids_all = [(x+1) for x in range(pid_embeddings_all.shape[0])]
    pids_all = np.array(pids_all)
    tree = TreeInitialize(pid_embeddings_all, pids_all, balance_factor=args.balance_factor, 
                          leaf_factor=args.leaf_factor, sampled_doc_num=args.sampled_doc_num,
                          kmeans_pkg=args.kmeans_pkg)
    _ = tree.clustering_tree()
    save_object(tree,tree_path)

    ## save node_dict
    tree = load_object(tree_path)
    node_dict = {}
    node_queue = [tree.root]
    val = []
    while node_queue:
        current_node = node_queue.pop(0) 
        node_dict[current_node.val] = current_node
        for child in current_node.children:
            node_queue.append(child)
    print("node dict length")
    print(len(node_dict))
    print("leaf dict length")
    print(len(tree.leaf_dict))
    # save_object(node_dict,f"{output_path}/node_dict.pkl")

    ## save node_list
    tree = load_object(tree_path)
    root = tree.root
    node_list = _node_list(root)
    # save_object(node_list,f"{output_path}/node_list.pkl")

    ## pid2cluster
    for leaf in tree.leaf_dict:
        node = tree.leaf_dict[leaf]
        pids = node.pids
        for pid in pids:
            dict_label[pid] = str(node.val)
    df = pd.DataFrame.from_dict(dict_label, orient='index',columns=['labels'])
    df = df.reset_index().rename(columns = {'index':'pid'})
    # df.to_csv(f"{output_path}/pid_labelid.memmap",header=False, index=False)
    
    # save
    code_dict = {}
    for idx,data in df.iterrows():
        assert data['pid'] not in code_dict
        code_dict[data['pid']] = ','.join(data['labels'].split('_')[1:])
    json.dump(code_dict, open(os.path.join(args.code_save_dir, f"tree-code-{base_model}.json"), 'w'), indent = 4)

    print('end')
    tree = load_object(tree_path)
    print(len(tree.leaf_dict))
    # save_object(tree.leaf_dict,f'{output_path}/leaf_dict.pkl')
