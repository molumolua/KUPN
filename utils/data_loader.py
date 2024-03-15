import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp

import random as rd
from time import time
from collections import defaultdict
import warnings
import collections
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_raw_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

    return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes,n_raw_relations

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users # including items + users
    n_relations = max(triplets[:, 1]) + 1 #考虑了正反关系

    return triplets


def build_graph(train_data, triplets):
    #正反图都有
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1]  # [0, n_items) -> [n_users, n_users+n_items) remap item
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # # interaction: user->item, [n_users, n_entities]
    # norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    # mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list

def get_all_kg_dict(triplets):
    all_kg_dict = collections.defaultdict(list)
    for h,r,t in triplets:
        all_kg_dict[h].append((t,r))
    return all_kg_dict

def generate_train_kg_batch(all_kg_dict,batch_size_kg,n_entities):
    exist_heads = all_kg_dict.keys()

    if batch_size_kg <= len(exist_heads):
        heads = rd.sample(exist_heads, batch_size_kg)
    else:
        heads = [rd.choice(exist_heads) for _ in range(batch_size_kg)]
    def sample_pos_triples_for_h(h, num):
        pos_triples = all_kg_dict[h]
        n_pos_triples = len(pos_triples)

        pos_rs, pos_ts = [], []
        while True:
            if len(pos_rs) == num: break
            pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            t = pos_triples[pos_id][0]
            r = pos_triples[pos_id][1]

            if r not in pos_rs and t not in pos_ts:
                pos_rs.append(r)
                pos_ts.append(t)
        return pos_rs, pos_ts

    def sample_neg_triples_for_h(h, r, num):
        neg_ts = []
        while True:
            if len(neg_ts) == num: break

            t = np.random.randint(low=0, high=n_entities, size=1)[0]
            if (t, r) not in all_kg_dict[h] and t not in neg_ts:
                neg_ts.append(t)
        return neg_ts
        
    pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

    for h in heads:
        pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
        pos_r_batch += pos_rs
        pos_t_batch += pos_ts

        neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
        neg_t_batch += neg_ts

    return heads, pos_r_batch, pos_t_batch, neg_t_batch


def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(directory + 'kg_final.txt')

    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)

    print('building the adj mat ...')
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    print("generate kg dict....")
    all_kg_dict=get_all_kg_dict(triplets)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations),
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph, \
           [adj_mat_list, norm_mat_list, mean_mat_list],all_kg_dict

