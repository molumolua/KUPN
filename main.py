import random

import torch
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import networkx as nx

import random
from time import time
from collections import defaultdict

from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from modules.newKG import Recommender
from utils.evaluate import test
from utils.helper import early_stopping
import matplotlib.pyplot as plt
import random as rd
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES']='3,5,6,7'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):

    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict

def generate_train_cl_batch(exist_items,batch_size_cl):
    if batch_size_cl <= len(exist_items):
        items = rd.sample(exist_items, batch_size_cl)
    else:
        items_list = list(exist_items)
        items = [rd.choice(items_list) for _ in range(batch_size_cl)]
    return items


def find_user_entity_neigh(adj_mat_list,n_users,n_nodes):
    #返回一个 user-entity的邻接矩阵列表
    tot=0
    inter_adj = adj_mat_list[0].copy()
    # print("min inter_adj_col:",min(inter_adj.col))
    inter_adj = inter_adj.tocsr()
    return_adj_mat_list = []
    for r_id,adj in enumerate(adj_mat_list):
        now_adj=None
        if(r_id==0):
            now_adj=adj.copy()
            # now_adj=sp.coo_matrix((adj.data, (adj.row, adj.col)), shape=adj.shape)
        else:
            knowleadge_adj = adj.tocsr()
            now_adj=inter_adj @ knowleadge_adj #user 和 item --》entity
            now_adj=now_adj.tocoo()
            # now_adj.col+=n_users  #remap entities

        # now_adj=sp.coo_matrix((now_adj.data, (now_adj.row, now_adj.col+n_users)), shape=now_adj.shape)
        now_adj.col+=n_users  # remap entities
        count=now_adj.data.size
        tot+=count
        # ''' show user entity information'''
        # print("latent relation : "+str(r_id)+" count: "+str(count))
        # values, counts = np.unique(now_adj.data, return_counts=True)
        # plt.figure(figsize=(10, 6))
        # plt.scatter(values, counts)
        # plt.xlabel('Value')
        # plt.ylabel('Times')
        # plt.title("Relation "+str(r_id))
        # plt.grid(True)
        # file_path = './pic/relation_'+str(r_id)+".png"
        # plt.savefig(file_path)


        # need test        
        # now_adj.data = np.ones_like(now_adj.data)
        if(count==0):
            continue

        assert max(now_adj.col) <n_nodes
        assert min(now_adj.col) >=n_users
        return_adj_mat_list.append(now_adj)
    print("total prefer relations:",tot)
    new_coo_matrices = []
    for matrix in return_adj_mat_list:
        # 交换行列来创建新的coo_matrix
        new_matrix = sp.coo_matrix((matrix.data, (matrix.col, matrix.row)), shape=matrix.shape)
        new_coo_matrices.append(new_matrix)
    return_adj_mat_list +=new_coo_matrices

    return return_adj_mat_list

def build_prefer_graph(adj_mat_list):
    prefer_graphs=[]
    for r_id,adj in enumerate(adj_mat_list):
        prefer_graph = nx.MultiDiGraph()
        for h_id, t_id, v in zip(adj.row, adj.col, adj.data):
            prefer_graph.add_edge(h_id, t_id, key=r_id)
        prefer_graphs.append(prefer_graph)

    return prefer_graphs




if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list,all_kg_dict = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """get user --item -- entity"""
    user_entity_mat_list=find_user_entity_neigh(adj_mat_list,n_users,n_nodes)
    prefer_graphs=build_prefer_graph(user_entity_mat_list)
    exist_nodes=[i for i in range(n_items+n_users)] #cl部分
    n_params['n_prefers']=len(user_entity_mat_list)

    print("n_user:",n_users)
    print("n_items:",n_items)
    print("n_entities:",n_entities)
    print("n_relations:",n_relations)
    print("n_nodes:",n_nodes)
    print("n_prefers:",n_params['n_prefers'])
    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))


    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0],prefer_graphs).to(device)


    # model = Recommender(n_params, args, graph, mean_mat_list[0],prefer_graph)
    # model = torch.nn.DataParallel(model)
    # model.to('cuda')


    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # cl_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print("start training ...")
    for epoch in range(args.epoch):
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        # model=model.train()
        '''test'''
        # model=model.eval()
        # test_s_t = time()
        # ret = test(model, user_dict, n_params)
        # test_e_t = time()

        # train_res = PrettyTable()
        # train_res.field_names = ["Epoch",  "tesing time", "recall", "ndcg", "precision", "hit_ratio"]
        # train_res.add_row(
        #     [epoch, test_e_t - test_s_t,ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
        # )
        # print(train_res)



        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]

        """training cf"""
        loss, s= 0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):

            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'])
            batch_loss, _, _ = model(batch)

            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

        train_e_t = time()

        """training cl"""
        s=0
        train_cl_s = time()
        cl_loss=0
        while s+args.batch_size_cl <= n_users+n_items:
            batch = generate_train_cl_batch(exist_nodes,args.batch_size_cl)

            batch_loss =model.get_cl_loss(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            cl_loss += batch_loss
            s += args.batch_size_cl

        train_cl_e = time()
        print("train cl time:",train_cl_e-train_cl_s)
        if epoch % 1 == 0 :
            """testing"""
            # model=model.eval()
            test_s_t = time()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss","CL_Loss","recall", "ndcg", "precision", "hit_ratio"]
            train_res.add_row(
                [epoch, train_cl_e - train_s_t, test_e_t - test_s_t, loss.item(), cl_loss.item(),ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
            )
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=20)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f,  epoch %d: training loss: %.4f  cl loss: %.4f' % (train_cl_e - train_s_t, epoch, loss.item(),cl_loss.item()))

    print('early stopping at %d, recall@10:%.4f' % (epoch, cur_best_pre_0))
