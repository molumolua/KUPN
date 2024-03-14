import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.scatter import scatter_mean
class Contrast_2view(nn.Module):
    def __init__(self, cf_dim, kg_dim, hidden_dim, tau, cl_size):
        super(Contrast_2view, self).__init__()
        self.projcf = nn.Sequential(
            nn.Linear(cf_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.projkg = nn.Sequential(
            nn.Linear(kg_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos = torch.eye(cl_size)
        self.tau = tau
        for model in self.projcf:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.projkg:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        sim_matrix = sim_matrix/(torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
        assert sim_matrix.size(0) == sim_matrix.size(1)
        lori_mp = -torch.log(sim_matrix.mul(self.pos.to(sim_matrix.device)).sum(dim=-1)).mean()
        return lori_mp

    def forward(self, z1, z2):
        multi_loss = False
        z1_proj = self.projcf(z1)
        z2_proj = self.projkg(z2)
        if multi_loss:
            loss1 = self.sim(z1_proj, z2_proj)
            loss2 = self.sim(z1_proj, z1_proj)
            loss3 = self.sim(z2_proj, z2_proj)
            return (loss1 + loss2 + loss3) / 3
        else:
            return self.sim(z1_proj, z2_proj)


class DropLearner(nn.Module):
    def __init__(self, node_dim, mlp_edge_model_dim = 64):
        super(DropLearner, self).__init__()
        
        self.mlp_src = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_dst = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_edge = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )

        self.mlp_con = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        
        self.concat = True
        

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, edge_index,edge_type,all_embed,relation_emb, temperature = 0.5):
        # print(relation_emb.shape)
        # print(torch.max(edge_type))
 
 

        head_emb=all_embed[edge_index[0,:]]
        tail_emb=all_embed[edge_index[1,:]]
        latent_emb=relation_emb[edge_type]

        # print(head_emb.shape)
        # print(tail_emb.shape)
        # print(latent_emb.shape)
        weight = self.mlp_con(head_emb + tail_emb + latent_emb)
        w_src = self.mlp_src(head_emb)
        w_dst = self.mlp_dst(tail_emb)
        w_edge = self.mlp_edge(latent_emb)
        weight += w_src + w_dst + w_edge
        weight = weight.squeeze()



        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(weight.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(head_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

        

        return aug_edge_weight
    

        # edge_drop_out_prob = 1 - aug_edge_weight
        # random_probs = torch.rand(edge_drop_out_prob.shape).to(head_emb.device)
        # # 比较随机数组与 edge_drop_out_prob 来决定是否保留每条边
        # edges_to_keep = random_probs > edge_drop_out_prob
        # kept_edge_index = edge_index[:,edges_to_keep]
        # kept_edge_type = edge_type[edges_to_keep]


        # return kept_edge_index,kept_edge_type
    

class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users,n_relations,n_nodes):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_nodes = n_nodes
        self.n_relations=n_relations
    
    # def forward(self,all_emb,edge_index,edge_type,weight,aug_edge_weight=None):
    #     """aggregate"""
    #     dim=all_emb.shape[0]
    #     head, tail = edge_index
    #     edge_relation_emb = weight[edge_type]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
    #     neigh_relation_emb = all_emb[tail] * edge_relation_emb  # [-1, channel]
    #     if aug_edge_weight is not None:
    #         neigh_relation_emb = neigh_relation_emb*aug_edge_weight
    #     res_emb = scatter_mean(src=neigh_relation_emb, index=head, dim_size=dim, dim=0)
    #     return res_emb
    

    def forward(self, entity_emb, user_emb,  #n种隐关系向量  [n_relations,latend_dim]
                edge_index, edge_type, extra_edge_index, extra_edge_type,  #替换成二阶+一阶 n_relations个矩阵   [n_relations,n_users,n_nodes]
                weight,extra_weight,aug_edge_weight=None,aug_extra_edge_weight=None):

        n_entities = entity_emb.shape[0]
        # channel = entity_emb.shape[1]
        # n_users = self.n_users
        n_nodes = self.n_nodes
        # n_relations=self.n_relations

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        if aug_edge_weight is not None:
            neigh_relation_emb = neigh_relation_emb*aug_edge_weight
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        # """cul user->latent factor attention"""
        # score_ = torch.mm(user_emb, latent_emb.t())
        # score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_relations, 1]

        # """user aggregate"""
        # user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        # disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),
        #                         weight).expand(n_users, n_relations, channel)
        # user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]

        """user prefer view aggregate"""
        all_embed= torch.concat([user_emb,entity_emb],dim=0)
        extra_head, extra_tail = extra_edge_index
        extra_edge_relation_emb = extra_weight[extra_edge_type]  #prefer
        extra_neigh_relation_emb = all_embed[extra_tail] * extra_edge_relation_emb  # [-1, channel]
        if aug_extra_edge_weight is not None:
            extra_neigh_relation_emb =extra_neigh_relation_emb*aug_extra_edge_weight

        node_agg = scatter_mean(src=extra_neigh_relation_emb, index=extra_head, dim_size=n_nodes, dim=0)
        return entity_agg, node_agg



class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                  n_relations, n_nodes,n_prefers,interact_mat,
                  node_dropout_rate=0.5, mess_dropout_rate=0.1,tau_prefer=1.5,tau_kg=1.5):
        super(GraphConv, self).__init__()
        #channel ---> embedding size
        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users

        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.tau_prefer=tau_prefer
        self.tau_kg=tau_kg

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        # 关系的embedding
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        # user-entity
        extra_weight = initializer(torch.empty(n_prefers, channel))  # not include interact
        self.extra_weight = nn.Parameter(extra_weight)  # [n_relations - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users,n_relations=n_relations,n_nodes=n_nodes))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

        self.drop_learner1 = DropLearner(channel)
        self.drop_learner2 = DropLearner(channel)

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_emb, entity_emb,  interact_mat,edge_index, edge_type,
               extra_edge_index,extra_edge_type,mess_dropout=True, node_dropout=False,drop_learn=False):
        # edge_index=edge_index.to(user_emb.device)
        # edge_type=edge_type.to(user_emb.device)
        # extra_edge_index=extra_edge_index.to(user_emb.device)
        # extra_edge_type=extra_edge_type.to(user_emb.device)
        # interact_mat=interact_mat.to(user_emb.device)

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            extra_edge_index, extra_edge_type = self._edge_sampling(extra_edge_index, extra_edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)
        
        aug_edge_weight,aug_extra_edge_weight=None,None

        # if drop_learn:
        #     aug_edge_weight=self.drop_learner1(edge_index,edge_type-1,entity_emb,\
        #                                             self.weight, temperature = self.tau_kg)
            
        #     aug_extra_edge_weight = self.drop_learner2(extra_edge_index, extra_edge_type,torch.concat([user_emb,entity_emb],dim=0), \
        #                                                            self.extra_weight, temperature = self.tau_prefer)
        #     aug_edge_weight = aug_edge_weight.unsqueeze(-1)
        #     aug_extra_edge_weight=aug_extra_edge_weight.unsqueeze(-1)
        

        entity_res_emb = entity_emb                               # [n_entity, channel]
        
        node_emb  = torch.concat([user_emb,entity_emb],dim=0)     # [n_nodes, channel]
        node_res_emb = node_emb

        user_res_emb = user_emb

        for i in range(len(self.convs)):
            # #all_emb,edge_index,edge_type,weight,aug_edge_weight=None
            # entity_emb = self.convs[i](entity_emb,edge_index,edge_type-1,self.weight,aug_edge_weight)
            # node_emb = self.convs[i](node_emb,extra_edge_index,extra_edge_type,self.extra_weight,aug_extra_edge_weight)
            entity_emb, node_emb = self.convs[i](entity_emb, node_emb[:self.n_users], 
                                                 edge_index, edge_type,extra_edge_index, extra_edge_type,
                                                 self.weight,self.extra_weight,
                                                 aug_edge_weight,aug_extra_edge_weight)
            

            user_emb =torch.sparse.mm(interact_mat,entity_emb)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                node_emb = self.dropout(node_emb)
                user_emb = self.dropout(user_emb)

            entity_emb = F.normalize(entity_emb)
            node_emb = F.normalize(node_emb)
            user_emb = F.normalize(user_emb)


            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            node_res_emb = torch.add(node_res_emb, node_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        gcn_res_emb=torch.concat([user_res_emb,entity_emb],dim=0) 
        return gcn_res_emb,node_res_emb
    


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat,extra_graphs):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.n_prefers = data_config['n_prefers']

        self.decay = args_config.l2

        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops #卷积层数
        self.cl_alpha=args_config.cl_alpha

        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate


        self.tau_prefer=args_config.tau_prefer
        self.tau_kg=args_config.tau_kg
        self.tau_cl=args_config.tau_cl
        self.keep_rate=args_config.keep_rate

        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        
        # self.device = torch.device("cuda" ) if args_config.cuda \
        #                                                               else torch.device("cpu")

        
        # self.adj_mat = adj_mat
        # self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        # self.extra_graph = extra_graph
        self.extra_edge_indexs = self._get_extra_edges(extra_graphs)

        self._init_weight(adj_mat)
        self.all_embed = nn.Parameter(self.all_embed)
        # self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()
        self.contrast1 = Contrast_2view(self.emb_size, self.emb_size, self.emb_size, self.tau_cl, args_config.batch_size_cl)
        # self.contrast2 = Contrast_2view(self.emb_size, self.emb_size, self.emb_size, self.tau_cl, args_config.batch_size_cl)

    def _init_weight(self,adj_mat):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        # self.latent_emb = initializer(torch.empty(self.n_relations, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(adj_mat).to(self.device)


    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_nodes=self.n_nodes,
                         n_prefers=self.n_prefers,
                         interact_mat=self.interact_mat,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                         tau_kg=self.tau_kg,
                         tau_prefer=self.tau_prefer)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col-self.n_users])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, [self.n_users,self.n_entities])

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _get_extra_edges(self,graphs):
        indexs=[]
        for graph in graphs:
            graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
            index = graph_tensor[:, :-1]  # [-1, 2]
            indexs.append(index.t().long().to(self.device))
        return indexs
    
    def _select_edges(self,indexs,keep_rate):
        select_indexs=[]
        select_types=[]
        for itype,index in enumerate(indexs):
            if itype ==0 or itype*2==self.n_prefers:
                random_numbers =torch.full([index.size(1)],0)
            else:
                random_numbers = torch.rand(index.size(1))
            # 根据 keep_rate 确定哪些行会被保留
            mask = random_numbers < keep_rate

            left_index=index[:,mask]
            if(left_index.shape[1]>0):
                left_type=torch.full([left_index.shape[1]],itype)
                select_indexs.append(left_index)
                select_types.append(left_type)
        return torch.concat(select_indexs,dim=1).to(self.device),torch.concat(select_types,dim=0).to(self.device)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.keep_rate)
        # print("extra_edge_index:",extra_edge_type.shape)
        # print("extra_edge_tpye:",extra_edge_type.shape)
        node_gcn_emb, node_prefer_emb =     self.gcn(user_emb,
                                                     item_emb,
                                                     self.interact_mat,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     extra_edge_index,
                                                     extra_edge_type,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout,
                                                     drop_learn=True)
        user_prefer_emb=node_prefer_emb[:self.n_users]
        entity_prefer_emb=node_prefer_emb[self.n_users:]

        user_gcn_emb=node_gcn_emb[:self.n_users]
        entity_gcn_emb=node_gcn_emb[self.n_users:]


        user_res_emb=torch.concat([user_gcn_emb,user_prefer_emb],dim=1)
        entity_res_emb=torch.concat([entity_gcn_emb,entity_prefer_emb],dim=1)


        u_e = user_res_emb[user]
        pos_e, neg_e = entity_res_emb[pos_item], entity_res_emb[neg_item]


        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def get_cl_loss(self,batch_nodes):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.keep_rate)

        node_gcn_emb, node_prefer_emb =     self.gcn(user_emb,
                                                     item_emb,
                                                     self.interact_mat,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     extra_edge_index,
                                                     extra_edge_type,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout,
                                                     drop_learn=True)
        # user_prefer_emb=node_prefer_emb[:self.n_users]
        # entity_prefer_emb=node_prefer_emb[self.n_users:]

        batch_gcn_emb =node_gcn_emb[batch_nodes]
        batch_prefer_emb=node_prefer_emb[batch_nodes]
        cl_loss = self.contrast1(batch_gcn_emb, batch_prefer_emb)
        loss = self.cl_alpha*cl_loss
        return loss
    
    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.keep_rate)
        node_gcn_emb, node_prefer_emb =     self.gcn(user_emb,
                                                     item_emb,
                                                     self.interact_mat,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     extra_edge_index,
                                                     extra_edge_type,
                                                     mess_dropout=False,
                                                     node_dropout=False,
                                                     drop_learn=True)
        
        user_prefer_emb=node_prefer_emb[:self.n_users]
        entity_prefer_emb=node_prefer_emb[self.n_users:]

        user_gcn_emb=node_gcn_emb[:self.n_users]
        entity_gcn_emb=node_gcn_emb[self.n_users:]


        user_res_emb=torch.concat([user_gcn_emb,user_prefer_emb],dim=1)
        entity_res_emb=torch.concat([entity_gcn_emb,entity_prefer_emb],dim=1)

        return entity_res_emb,user_res_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss
