import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from utils.scatter import scatter_mean,scatter_sum
from torch_geometric.utils import softmax as scatter_softmax
import math
def prefix_product(lst):
    
    result = [lst[0]]  # 初始值设置为列表的第一个元素
    for num in lst[1:]:  # 从第二个元素开始遍历
        result.append(result[-1] * num)  # 将前一个结果与当前元素相乘，并追加到结果列表
    return result


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
    def __init__(self, node_dim, n_relations,mlp_edge_model_dim = 64):
        super(DropLearner, self).__init__()
        
        # self.mlp_src = nn.Sequential(
        #     nn.Linear(node_dim, mlp_edge_model_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_edge_model_dim, 1)
        # )
        # self.mlp_dst = nn.Sequential(
        #     nn.Linear(node_dim, mlp_edge_model_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_edge_model_dim, 1)
        # )
        # self.mlp_edge = nn.Sequential(
        #     nn.Linear(node_dim, mlp_edge_model_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_edge_model_dim, 1)
        # )

        self.mlp_con = nn.Sequential(
            nn.Linear(3*node_dim, 3*mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(3*mlp_edge_model_dim, 1)
        )
        
        # self.concat = True
        

        self.init_emb()
        # self.n_relations=n_relations
        # self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, node_dim, node_dim))
        # nn.init.xavier_uniform_(self.trans_M)

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
        # W_r = self.trans_M[edge_type]

        # r_mul_h = torch.bmm(head_emb.unsqueeze(1), W_r).squeeze(1)
        # r_mul_t = torch.bmm(tail_emb.unsqueeze(1), W_r).squeeze(1)

        # print(head_emb.shape)
        # print(tail_emb.shape)
        # print(latent_emb.shape)
        # weight = self.mlp_con(head_emb + tail_emb + latent_emb)

        # w_src = self.mlp_src(head_emb)
        # w_dst = self.mlp_dst(tail_emb)
        # w_edge = self.mlp_edge(latent_emb)
        # weight = w_src + w_dst + w_edge

        weight = self.mlp_con(torch.concat([head_emb, tail_emb ,latent_emb],dim=1))
        weight = weight.squeeze()


        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(weight.size()) + (1 - bias)   # 1-rand()
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
    def __init__(self, n_users,n_relations,n_prefers,n_nodes,channel):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_nodes = n_nodes
        self.n_relations=n_relations
        self.n_prefers=n_prefers

        self.n_heads=2
        self.d_k=channel//self.n_heads


    
    def forward(self,all_emb,edge_index,edge_type,weight,aug_edge_weight=None,div=False,with_relation=True):
        """aggregate"""
        dim=all_emb.shape[0]
        head, tail = edge_index
        if with_relation:
            edge_relation_emb = weight[edge_type]
            neigh_relation_emb = all_emb[tail] * edge_relation_emb  # [-1, channel]
        else :
            neigh_relation_emb = all_emb[tail]

        if aug_edge_weight is not None:
            # neigh_relation_emb = neigh_relation_emb.view(-1, self.n_heads, self.d_k)*aug_edge_weight.view(-1, self.n_heads, 1)
            # neigh_relation_emb = neigh_relation_emb.view(-1,self.n_heads*self.d_k)
            neigh_relation_emb *=aug_edge_weight
        if div:
            res_emb = scatter_mean(src=neigh_relation_emb, index=head, dim_size=dim, dim=0)
        else:
            res_emb = scatter_sum(src=neigh_relation_emb, index=head,dim_size=dim,dim=0)
        return res_emb
    
    def batch_get_contribute(self,all_emb,edge_index,edge_type,weight,mask,batch_size,aug_edge_weight,rate,with_relation=True):
        dim = all_emb.shape[0]
        channel = all_emb.shape[1]
        head, tail = edge_index
        head=head[mask]
        tail=tail[mask]
        edge_type=edge_type[mask]
        aug_edge_weight=aug_edge_weight[mask]
        n_batches = (edge_index.shape[1] + batch_size - 1) // batch_size
        contrib_sum = torch.zeros(dim, channel).to(edge_index.device)
        degrees = torch.zeros(dim).to(edge_index.device)
        for b in range(n_batches):
            start_idx = b * batch_size
            # print(start_idx,edge_index.shape[1])
            end_idx = min((b + 1) * batch_size, edge_index.shape[1])

            head_batch = head[start_idx:end_idx]
            tail_batch = tail[start_idx:end_idx]

            if with_relation:
                edge_type_batch = edge_type[start_idx:end_idx]
                edge_relation_emb_batch = weight[edge_type_batch]
                neigh_relation_emb_batch = all_emb[tail_batch] * edge_relation_emb_batch
            else:
                neigh_relation_emb_batch = all_emb[tail_batch]

            if aug_edge_weight is not None:
                aug_edge_weight_batch = aug_edge_weight[start_idx:end_idx]
                neigh_relation_emb_batch *= aug_edge_weight_batch.unsqueeze(-1)
                # neigh_relation_emb_batch = neigh_relation_emb_batch.view(-1, self.n_heads, self.d_k)*aug_edge_weight_batch.view(-1, self.n_heads, 1)
                # neigh_relation_emb_batch = neigh_relation_emb_batch.view(-1,self.n_heads*self.d_k)
            # 累加当前批次的贡献
            contrib_sum.index_add_(0, head_batch, neigh_relation_emb_batch*rate)
            # 累加当前批次每个节点的出现次数
            degrees.index_add_(0, head_batch, torch.ones_like(head_batch, dtype=torch.float)*rate)
        return degrees,contrib_sum
        


    def batch_generate(self,all_emb,edge_index,edge_type,weight,prefers,aug_edge_weight=None,batch_size=16777216,zero_rate=1.0,div=False,with_relation=True):
        """aggregate"""
        # zero_mask=(edge_type == 0) | (edge_type == self.n_prefers/2) | (~torch.isin(edge_type, prefers))  #user --item or user -- path
        zero_mask=(edge_type == 0) | (edge_type == self.n_prefers/2)
        zero_degrees,zero_contrib_sum=self.batch_get_contribute(all_emb,edge_index,edge_type,weight,zero_mask,batch_size,aug_edge_weight,zero_rate,with_relation)


        nonzero_mask=~ zero_mask # user entity
        nonzero_degrees,nonzero_contrib_sum=self.batch_get_contribute(all_emb,edge_index,edge_type,weight,nonzero_mask,batch_size,aug_edge_weight,1.0,with_relation)
        
        degrees=nonzero_degrees+zero_degrees
        contrib_sum=nonzero_contrib_sum+zero_contrib_sum
        if div:
            degrees[degrees == 0] = 1
            res_emb = contrib_sum / degrees.unsqueeze(-1)
        else:
            res_emb = contrib_sum
        return res_emb

    

    

    # def forward(self, entity_emb, user_emb,  #n种隐关系向量  [n_relations,latend_dim]
    #             edge_index, edge_type, extra_edge_index, extra_edge_type,  #替换成二阶+一阶 n_relations个矩阵   [n_relations,n_users,n_nodes]
    #             weight,extra_weight,aug_edge_weight=None,aug_extra_edge_weight=None):

    #     n_entities = entity_emb.shape[0]
    #     # channel = entity_emb.shape[1]
    #     # n_users = self.n_users
    #     n_nodes = self.n_nodes
    #     # n_relations=self.n_relations

    #     """KG aggregate"""
    #     head, tail = edge_index
    #     edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
    #     neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
    #     if aug_edge_weight is not None:
    #         neigh_relation_emb = neigh_relation_emb*aug_edge_weight
    #     entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

    #     # """cul user->latent factor attention"""
    #     # score_ = torch.mm(user_emb, latent_emb.t())
    #     # score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_relations, 1]

    #     # """user aggregate"""
    #     # user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
    #     # disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),
    #     #                         weight).expand(n_users, n_relations, channel)
    #     # user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]

    #     """user prefer view aggregate"""
    #     all_embed= torch.concat([user_emb,entity_emb],dim=0)
    #     extra_head, extra_tail = extra_edge_index
    #     extra_edge_relation_emb = extra_weight[extra_edge_type]  #prefer
    #     extra_neigh_relation_emb = all_embed[extra_tail] * extra_edge_relation_emb  # [-1, channel]
    #     if aug_extra_edge_weight is not None:
    #         extra_neigh_relation_emb =extra_neigh_relation_emb*aug_extra_edge_weight

    #     node_agg = scatter_mean(src=extra_neigh_relation_emb, index=extra_head, dim_size=n_nodes, dim=0)
    #     return entity_agg, node_agg



class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                  n_relations, n_nodes,n_prefers,interact_mat,
                  node_dropout_rate=0.5, mess_dropout_rate=0.1,device=None):
        super(GraphConv, self).__init__()
        #channel ---> embedding size
        self.device=device
        self.convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users

        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate


        initializer = nn.init.xavier_uniform_
        # 关系的embedding
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        # user-entity
        extra_weight = initializer(torch.empty(n_prefers, channel))
        self.extra_weight = nn.Parameter(extra_weight)  # [n_relations - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users,n_relations=n_relations,n_prefers=n_prefers,n_nodes=n_nodes,channel=channel))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

        # self.drop_learner1 = DropLearner(channel,n_relations)
        # self.drop_learner2 = DropLearner(channel,n_prefers)

        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))

        self.W_K = nn.Parameter(torch.Tensor(channel, channel))
        
        self.n_heads = 2
        self.d_k = channel // self.n_heads

        nn.init.xavier_uniform_(self.W_Q)
        # nn.init.xavier_normal_(self.W_Q, gain=0.2)


        nn.init.xavier_uniform_(self.W_K)
        # self.W_K = nn.Parameter(torch.zeros(channel, channel))
        # nn.init.xavier_uniform_(self.W_K, gain=0.01)
        # # nn.init.uniform_(self.W_K, a=-0.01, b=0.01)
        # nn.init.normal_(self.W_K, mean=0, std=0.01)



    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _edge_sampling_torch(self,edge_index, edge_type, rate=0.5):
        n_edges = edge_index.size(1)
        # 使用torch.rand生成一个[0, 1)区间的随机张量，然后选择小于给定采样率的索引
        mask = torch.rand(n_edges, device=edge_index.device) < rate
        # 应用掩码选择索引
        sampled_edge_index = edge_index[:, mask]
        sampled_edge_type = edge_type[mask]
        return sampled_edge_index, sampled_edge_type


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
               extra_edge_index,extra_edge_type,mess_dropout=True, node_dropout=False,drop_learn=False,method="add"):
        # edge_index=edge_index.to(user_emb.device)
        # edge_type=edge_type.to(user_emb.device)
        # extra_edge_index=extra_edge_index.to(user_emb.device)
        # extra_edge_type=extra_edge_type.to(user_emb.device)
        # interact_mat=interact_mat.to(user_emb.device)

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling_torch(edge_index, edge_type, self.node_dropout_rate)
            extra_edge_index, extra_edge_type = self._edge_sampling_torch(extra_edge_index, extra_edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)
        aug_edge_weight,aug_extra_edge_weight=None,None
        node_emb  = torch.concat([user_emb,entity_emb],dim=0)     # [n_nodes, channel]

        # if drop_learn:
        #     aug_edge_weight=self.drop_learner1(edge_index,edge_type-1,entity_emb,
        #                                             self.weight, temperature = self.tau_kg)
            
        #     aug_extra_edge_weight = self.drop_learner2(extra_edge_index, extra_edge_type,node_emb, 
        #                                                            self.extra_weight, temperature = self.tau_prefer)
        #     aug_edge_weight = aug_edge_weight.unsqueeze(-1)
        #     aug_extra_edge_weight=aug_extra_edge_weight.unsqueeze(-1)
        

        aug_extra_edge_weight=self.calc_attn_score(node_emb,extra_edge_index,extra_edge_type).unsqueeze(-1)

        entity_res_emb = entity_emb                               # [n_entity, channel]
        
        node_res_emb = node_emb
        if method == "add":
            user_res_emb = user_emb
        elif method =="stack":
            user_res_emb = [user_emb]
        else:
            raise NotImplementedError


        for i in range(len(self.convs)):
            #all_emb,edge_index,edge_type,weight,aug_edge_weight=None
            entity_emb = self.convs[i](entity_emb,edge_index,edge_type-1,self.weight,aug_edge_weight,div=True)
            node_emb = self.convs[i](node_emb,extra_edge_index,extra_edge_type,self.extra_weight,
                                     aug_extra_edge_weight,div=True,with_relation=False)
            # entity_emb, node_emb = self.convs[i](entity_emb, node_emb[:self.n_users], 
            #                                      edge_index, edge_type,extra_edge_index, extra_edge_type,
            #                                      self.weight,self.extra_weight,
            #                                      aug_edge_weight,aug_extra_edge_weight)
            

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
            if method == "add":
                user_res_emb = torch.add(user_res_emb, user_emb)
            elif method =="stack":
                user_res_emb +=[user_emb]
            else:
                raise NotImplementedError
            
        if method == "stack":
            user_res_emb =torch.stack(user_res_emb,dim=1)
            user_res_emb =torch.mean(user_res_emb,dim=1)


        gcn_res_emb=torch.concat([user_res_emb,entity_res_emb],dim=0) 
        return gcn_res_emb,node_res_emb
    
    def batch_generate(self, user_emb, entity_emb,  interact_mat,edge_index, edge_type,
               extra_edge_index,extra_edge_type,method,keep_rate,prefers):
        
        node_emb  = torch.concat([user_emb,entity_emb],dim=0)
        entity_res_emb = entity_emb                               # [n_entity, channel]
        node_res_emb = node_emb
        if method == "add":
            user_res_emb = user_emb
        elif method =="stack":
            user_res_emb = [user_emb]
        else:
            raise NotImplementedError

        # user_res_emb = [user_emb]
        with torch.no_grad():
            aug_extra_edge_weight=self.batch_calc_attn_score(node_emb,extra_edge_index,extra_edge_type)
            if keep_rate ==0:
                aug_extra_edge_weight=aug_extra_edge_weight.unsqueeze(-1)
            for i in range(len(self.convs)):
                #all_emb,edge_index,edge_type,weight,aug_edge_weight=None
                entity_emb = self.convs[i](entity_emb,edge_index,edge_type-1,self.weight,div=True)
                if keep_rate == 0:
                    node_emb = self.convs[i](node_emb,extra_edge_index,extra_edge_type,self.extra_weight,
                                     aug_extra_edge_weight,div=True,with_relation=False)
                else:
                    node_emb = self.convs[i].batch_generate(node_emb,extra_edge_index,extra_edge_type,self.extra_weight,
                                                            aug_edge_weight=aug_extra_edge_weight,
                                                            prefers=torch.tensor(prefers).to(node_emb.device),
                                                            zero_rate=1.0/keep_rate,
                                                            div=True,
                                                            with_relation=False)
                
                user_emb =torch.sparse.mm(interact_mat,entity_emb)

                entity_emb = F.normalize(entity_emb)
                node_emb = F.normalize(node_emb)
                user_emb = F.normalize(user_emb)
            


                """result emb"""
                entity_res_emb = torch.add(entity_res_emb, entity_emb)
                node_res_emb = torch.add(node_res_emb, node_emb)
                if method == "add":
                    user_res_emb = torch.add(user_res_emb, user_emb)
                elif method =="stack":
                    user_res_emb +=[user_emb]
                else:
                    raise NotImplementedError
            if method == "stack":
                user_res_emb =torch.stack(user_res_emb,dim=1)
                user_res_emb =torch.mean(user_res_emb,dim=1)

            gcn_res_emb=torch.concat([user_res_emb,entity_res_emb],dim=0) 

        return gcn_res_emb,node_res_emb


    # def calc_attn_score(self,all_emb,edge_index,edge_type):
    #     n_nodes = all_emb.shape[0]
    #     head, tail = edge_index

    #     query = (all_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
    #     key = (all_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)
    #     key = key * self.extra_weight[edge_type].view(-1, self.n_heads, self.d_k)

    #     edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
    #     # softmax by head_node
    #     edge_attn_score = scatter_softmax(edge_attn, head)
    #     # # normalization by head_node degree
    #     # norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=n_nodes)
    #     # norm = torch.index_select(norm, 0, head)
    #     # edge_attn_score = edge_attn_score * norm
        
    #     return edge_attn_score

    # @torch.no_grad()
    # def batch_calc_attn_score(self,all_emb,edge_index,edge_type,batch_size=1024):
    #     n_nodes = all_emb.shape[0]
    #     head, tail = edge_index

    #     n_batches = (edge_index.shape[1] + batch_size - 1) // batch_size
    #     edge_attns=[]
    #     for b in range(n_batches):
    #         start_idx = b * batch_size
    #         # print(start_idx,edge_index.shape[1])
    #         end_idx = min((b + 1) * batch_size, edge_index.shape[1])

    #         head_batch = head[start_idx:end_idx]
    #         tail_batch = tail[start_idx:end_idx]
    #         edge_type_batch = edge_type[start_idx:end_idx]

    #         query_batch = (all_emb[head_batch] @ self.W_Q).view(-1, self.n_heads, self.d_k)
    #         key_batch = (all_emb[tail_batch] @ self.W_Q).view(-1, self.n_heads, self.d_k)
    #         key_batch = key_batch * self.extra_weight[edge_type_batch].view(-1, self.n_heads, self.d_k)
    #         edge_attn_batch = (query_batch * key_batch).sum(dim=-1) / math.sqrt(self.d_k)
    #         edge_attns.append(edge_attn_batch)

    #     edge_attn =torch.concat(edge_attns,dim=0)
    #     # softmax by head_node
    #     edge_attn_score = scatter_softmax(edge_attn, head)
    #     # # normalization by head_node degree
    #     # norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=n_nodes)
    #     # norm = torch.index_select(norm, 0, head)
    #     # edge_attn_score = edge_attn_score * norm

    #     # print(edge_attn_score.shape)
    #     return edge_attn_score

    def calc_attn_score_core(self,all_emb,edge_type,head,tail):
        h_r = all_emb[head] @ self.W_Q
        h_r = h_r * self.extra_weight[edge_type]
        t_r = all_emb[tail]  @ self.W_K
        t_r = t_r * self.extra_weight[edge_type]


        h_r = h_r/torch.norm(h_r,dim=1,keepdim=True)
        t_r = t_r/torch.norm(t_r,dim=1,keepdim=True)

        edge_attn = (h_r * t_r).sum(dim=-1)
        return edge_attn

    
    def calc_attn_score(self,all_emb,edge_index,edge_type):
        head, tail = edge_index
        edge_attn = self.calc_attn_score_core(all_emb,edge_type,head,tail)
        edge_attn_score = scatter_softmax(edge_attn, head)
        
        return edge_attn_score
    
    
    def batch_calc_attn_score(self,all_emb,edge_index,edge_type,batch_size=1024):
        n_nodes = all_emb.shape[0]
        head, tail = edge_index

        n_batches = (edge_index.shape[1] + batch_size - 1) // batch_size
        edge_attns=[]
        for b in range(n_batches):
            start_idx = b * batch_size
            # print(start_idx,edge_index.shape[1])
            end_idx = min((b + 1) * batch_size, edge_index.shape[1])

            head_batch = head[start_idx:end_idx]
            tail_batch = tail[start_idx:end_idx]
            edge_type_batch = edge_type[start_idx:end_idx]
            
            edge_attn=self.calc_attn_score_core(all_emb,edge_type_batch,head_batch,tail_batch)
            edge_attns.append(edge_attn)

        edge_attn =torch.concat(edge_attns,dim=0)
        # softmax by head_node
        edge_attn_score = scatter_softmax(edge_attn, head)

        return edge_attn_score
    
    @torch.no_grad()
    def find_high_order_neigh(self,all_emb,extra_edge_index,extra_edge_type,head_dict,n_users,n_relations,batch_size=None,Ks=[2,1]):
        if len(Ks)<=1:
            return None,None
        high_order_set=set()    
        mask = (extra_edge_index[0] <=n_users) & (extra_edge_type  >0)

        extra_edge_index=extra_edge_index[:,mask]
        extra_edge_type =extra_edge_type [mask]


        if batch_size is not None:
            edge_attn_score=self.batch_calc_attn_score(all_emb,extra_edge_index,extra_edge_type,batch_size)
        else:
            edge_attn_score=self.calc_attn_score(all_emb,extra_edge_index,extra_edge_type)
        unique_users,topk_entities=self.find_topk_entities(extra_edge_index,edge_attn_score,Ks[0],max(Ks))


        back_entities=topk_entities
        nxt_entities = torch.full((len(unique_users), max(Ks)+1), -1, dtype=torch.long)  # 使用-1作为填充值
        indexs=[]
        types=[]
        l = len(Ks[1:])
        for  ik,K in enumerate(Ks[1:]):
            for i,u in enumerate(unique_users):
                now_head=torch.tensor(u)
                now_entities=[]
                cnt=0 #user 这一阶上的所有的entity
                for entity in back_entities[i]:
                    if entity == -1:
                        break
                    now_list=head_dict[entity.item()]
                    if len(now_list)>0:
                        entities_back = random.sample(now_list,min(3*K,len(head_dict[entity.item()])))
                        now_entities.extend(entities_back)
                        
                if len(now_entities)>0:    
                    now_list=torch.tensor(now_entities).to(all_emb.device)
                    now_type=now_list[:,1]
                    now_tail=now_list[:,0]
                    attn_score=F.softmax(self.calc_attn_score_core(all_emb,edge_type=now_type,head=now_head,tail=now_tail))
                    # print("attn score:",attn_score)
                    noise = torch.randn_like(attn_score) * 0
                    noisy_scores =attn_score + noise
                    topk_vals, topk_indices = torch.topk(noisy_scores, k=K)
                    entity_nxt = now_tail[topk_indices].unsqueeze(-1)
                    type_nxt = now_type[topk_indices].unsqueeze(-1)
                    entities_nxt = torch.concat([entity_nxt,type_nxt],dim=1).tolist()
                    for e,r in entities_nxt:
                        indexs.append((u,e))
                        types.append((r))
                        indexs.append((e,u))
                        types.append((r+n_relations))
                        high_order_set.add(e)
                        if ik<l-1:  #0
                            nxt_entities[i][cnt]=e
                            cnt+=1
                nxt_entities[i][cnt]=-1

                # for entity in back_entities[i]:
                #     if entity == -1:
                #         break
                #     now_list=head_dict[entity.item()]

                #     if len(now_list)>0:
                #         now_list=torch.tensor(head_dict[entity.item()]).to(all_emb.device) 
                #         now_type=now_list[:,1]
                #         now_tail=now_list[:,0]
                #         attn_score=F.softmax(self.calc_attn_score_core(all_emb,edge_type=now_type,head=now_head,tail=now_tail))
                #         # print("attn score:",attn_score)
                #         noise = torch.randn_like(attn_score) * 0.1
                #         noisy_scores =attn_score + noise
                #         topk_vals, topk_indices = torch.topk(noisy_scores, k=K)
                #         entity_nxt = now_tail[topk_indices].unsqueeze(-1)
                #         type_nxt = now_type[topk_indices].unsqueeze(-1)
                #         entities_nxt = torch.concat([entity_nxt,type_nxt],dim=1).tolist()
                #         # now_list=head_dict[entity.item()]
                #         # entities_nxt = random.sample(now_list,min(K,len(head_dict[entity.item()])))
                #         for e,r in entities_nxt:
                #             indexs.append((u,e))
                #             types.append((r))
                #             indexs.append((e,u))
                #             types.append((r+n_relations))
                #             if ik<l-1:  #0
                #                 nxt_entities[i][cnt]=e
                #                 cnt+=1
                # nxt_entities[i][cnt]=-1

            back_entities=nxt_entities.clone()


        return torch.tensor(indexs).t().long().to(all_emb.device),torch.tensor(types).long().to(all_emb.device),high_order_set




    def select_low_order_neigh(self,all_emb,indexs,types,keep_rate):
        select_indexs=[]
        select_types=[]
        for i in range(len(indexs)):
            index=indexs[i]
            type=types[i]
            if type[0] ==0 or type[0]==self.n_relations:
                select_indexs.append(index)
                select_types.append(type)
                continue

            attn_score=self.calc_attn_score_core(all_emb,edge_type=type,head=index[:,0],tail=index[:,1])
            topk_vals, topk_indices = torch.topk(attn_score, k=int(keep_rate*index.shape[0]))
            topk_indices =topk_indices.to(index.device)

            left_index=index[topk_indices,:]
            left_type=type[topk_indices]
            if(left_index.shape[0]>0):
                select_indexs.append(left_index)
                select_types.append(left_type)

        return torch.concat(select_indexs,dim=0).t().to(self.device),torch.concat(select_types,dim=0).to(self.device)
    
    def find_topk_entities(self,edge_index, edge_scores, K,maxK):
        # edge_index的第一行包含用户索引
        user_indices = edge_index[0]
        
        # 按用户索引排序，保持注意力分数与排序同步
        sorted_indices = user_indices.argsort()
        sorted_user_indices = user_indices[sorted_indices]
        sorted_entity_indices = edge_index[1][sorted_indices]
        sorted_scores = edge_scores[sorted_indices]
        
        unique_users, inverse_indices = torch.unique_consecutive(sorted_user_indices, return_inverse=True)

        # # 使用inverse_indices来计算每个唯一元素在原始数组中的起始索引
        counts = torch.bincount(inverse_indices)
        first_unique_indices = torch.zeros_like(counts).scatter_(0, torch.arange(len(counts)).to(counts.device), counts).cumsum(0) - counts


        # 为了获取每个用户的Top-K，初始化存储结果的列表
        topk_entities = torch.full((len(unique_users), maxK+1), -1, dtype=torch.long)  # 使用-1作为填充值
        
        # 遍历每个用户，获取Top-K
        for i, user in enumerate(unique_users):
            start_idx = first_unique_indices[i]
            end_idx = first_unique_indices[i + 1] if i + 1 < len(first_unique_indices) else len(sorted_user_indices)
            # print("start_idx:",start_idx)
            # print("end_idx:",end_idx)
            # 获取当前用户的得分及其对应的实体索引
            scores = sorted_scores[start_idx:end_idx]
            entities = sorted_entity_indices[start_idx:end_idx]
            
            # 提取Top-K得分及其对应实体的索引
            topk_scores, topk_indices = torch.topk(scores, min(K,scores.shape[0]), largest=True, sorted=True)
            
            # 存储Top-K实体
            actual_k = min(K, len(entities))  # 实际上可能存在的连接数目可能小于K
            topk_entities[i, :actual_k] = entities[topk_indices][:actual_k]
        
        return unique_users, topk_entities

        

class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat,extra_graphs,init_prefers=None):
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

        # self.drop_learn=args_config.drop_learn


        # self.tau_prefer=args_config.tau_prefer
        # self.tau_kg=args_config.tau_kg
        self.init_prefers=init_prefers
        self.tau_cl=args_config.tau_cl
        self.keep_rate=args_config.keep_rate
        self.method=args_config.method

        self.neighs=eval(args_config.neighs)


        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        
        # self.device = torch.device("cuda" ) if args_config.cuda \
        #                                                               else torch.device("cpu")

        
        # self.adj_mat = adj_mat
        # self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        # self.extra_graph = extra_graph
        self.extra_edge_indexs,self.extra_edge_types = self._get_extra_edges(extra_graphs)

        self._init_weight(adj_mat)
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
                         device=self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        # print("min index:",min(coo.col))
        i = torch.LongTensor([coo.row, coo.col])
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
        types=[]
        for graph in graphs:
            graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
            index = graph_tensor[:, :-1]  # [-1, 2]
            indexs.append(index.long().cpu())
            type = graph_tensor[:, -1]
            types.append(type.long().cpu())
        return indexs,types
    
    def _select_edges(self,indexs,types,keep_rate,extra_index=None,extra_type=None):
        select_indexs=[]
        select_types=[]
        for i in range(len(indexs)):
            index=indexs[i]
            type=types[i]
            if type[0] ==0 or type[0]==self.n_relations:
                random_numbers =torch.full([index.size(0)],0)
            else:
                random_numbers = torch.rand(index.size(0))
            # 根据 keep_rate 确定哪些行会被保留
            mask = random_numbers <= keep_rate

            left_index=index[mask,:]
            left_type=type[mask]
            # print("type:",itype,"chosen:",left_index.shape)
            if(left_index.shape[0]>0):
                select_indexs.append(left_index)
                select_types.append(left_type)
        if extra_index is not None and extra_type is not None:
            extra_index = extra_index.t().cpu()
            extra_type=extra_type.cpu()
            random_numbers = torch.rand(extra_index.size(0))
            mask = random_numbers <= keep_rate
            left_index=extra_index[mask,:]
            left_type=extra_type[mask]
            if(left_index.shape[0]>0):
                    select_indexs.append(left_index)
                    select_types.append(left_type)

        return torch.concat(select_indexs,dim=0).t().to(self.device),torch.concat(select_types,dim=0).to(self.device)

    def forward(self, batch=None,index_new=None,type_new=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        # extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.extra_edge_types,self.keep_rate,
        #                                                      index_new,type_new)

        extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.extra_edge_types,self.keep_rate)
        if index_new is not None and type_new is not None:
            extra_edge_index=torch.concat([extra_edge_index,index_new],dim=1)
            extra_edge_type=torch.concat([extra_edge_type,type_new],dim=0)

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
                                                     method=self.method)

        
        user_prefer_emb=node_prefer_emb[:self.n_users]
        entity_prefer_emb=node_prefer_emb[self.n_users:]

        user_gcn_emb=node_gcn_emb[:self.n_users]
        entity_gcn_emb=node_gcn_emb[self.n_users:]
        user_res_emb=torch.concat([user_gcn_emb,user_prefer_emb],dim=1)
        entity_res_emb=torch.concat([entity_gcn_emb,entity_prefer_emb],dim=1)

        u_e = user_res_emb[user]
        pos_e, neg_e = entity_res_emb[pos_item], entity_res_emb[neg_item]

        return self.create_bpr_loss(u_e, pos_e, neg_e)

    def get_cl_loss(self,batch_nodes,index_new=None,type_new=None):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        # extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.extra_edge_types,self.keep_rate)
        # if index_new is not None and type_new is not None:
        #     extra_edge_index=torch.concat([extra_edge_index,index_new],dim=1)
        #     extra_edge_type=torch.concat([extra_edge_type,type_new],dim=0)

        # # extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.extra_edge_types,self.keep_rate,
        # #                                                      index_new,type_new)

        # node_gcn_emb, node_prefer_emb =     self.gcn(user_emb,
        #                                              item_emb,
        #                                              self.interact_mat,
        #                                              self.edge_index,
        #                                              self.edge_type,
        #                                              extra_edge_index,
        #                                              extra_edge_type,
        #                                              mess_dropout=self.mess_dropout,
        #                                              node_dropout=self.node_dropout,
        #                                              method=self.method)
        # # user_prefer_emb=node_prefer_emb[:self.n_users]
        # # entity_prefer_emb=node_prefer_emb[self.n_users:]

        if self.keep_rate ==0:
            extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.extra_edge_types,0)
        else:
            extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.extra_edge_types,1)


        if index_new is not None and type_new is not None:
            extra_edge_index=torch.concat([extra_edge_index,index_new],dim=1)
            extra_edge_type=torch.concat([extra_edge_type,type_new],dim=0)

        # extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.extra_edge_types,1,
        #                                                  index_new,type_new)
            
        node_gcn_emb, node_prefer_emb =     self.gcn.batch_generate(user_emb,
                                                         item_emb,
                                                         self.interact_mat,
                                                         self.edge_index,
                                                         self.edge_type,
                                                         extra_edge_index,
                                                         extra_edge_type,
                                                         self.method,
                                                         self.keep_rate,
                                                         self.init_prefers)
        

        batch_gcn_emb =node_gcn_emb[batch_nodes]
        batch_prefer_emb=node_prefer_emb[batch_nodes]
        cl_loss = self.contrast1(batch_gcn_emb, batch_prefer_emb)
        loss = self.cl_alpha*cl_loss
        return loss
    
    def generate(self,index_new=None,type_new=None):
        with torch.no_grad():
            user_emb = self.all_embed[:self.n_users, :]
            item_emb = self.all_embed[self.n_users:, :]
            if self.keep_rate ==0:
                extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.extra_edge_types,0)
            else:
                extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.extra_edge_types,1)


            if index_new is not None and type_new is not None:
                extra_edge_index=torch.concat([extra_edge_index,index_new],dim=1)
                extra_edge_type=torch.concat([extra_edge_type,type_new],dim=0)

            # extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.extra_edge_types,1,
            #                                                  index_new,type_new)
            
            node_gcn_emb, node_prefer_emb =     self.gcn.batch_generate(user_emb,
                                                         item_emb,
                                                         self.interact_mat,
                                                         self.edge_index,
                                                         self.edge_type,
                                                         extra_edge_index,
                                                         extra_edge_type,
                                                         self.method,
                                                         self.keep_rate,
                                                         self.init_prefers)

            # extra_edge_index,extra_edge_type=self.gcn.select_low_order_neigh(self.all_embed,self.extra_edge_indexs,self.extra_edge_types,self.keep_rate)

            # if index_new is not None and type_new is not None:
            #     extra_edge_index=torch.concat([extra_edge_index,index_new],dim=1)
            #     extra_edge_type=torch.concat([extra_edge_type,type_new],dim=0)

            
            # node_gcn_emb, node_prefer_emb = self.gcn(user_emb,
            #                                          item_emb,
            #                                          self.interact_mat,
            #                                          self.edge_index,
            #                                          self.edge_type,
            #                                          extra_edge_index,
            #                                          extra_edge_type,
            #                                          mess_dropout=self.mess_dropout,
            #                                          node_dropout=self.node_dropout,
            #                                          method=self.method)
 
            
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
        # print("users:",users)
        # print("pos items:",pos_items)
        # print("neg items:",neg_items)
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        
        return mf_loss + emb_loss, mf_loss, emb_loss
    
    def find_high_order_neigh(self,head_dict,batch_size=None):
        extra_edge_index,extra_edge_type=self._select_edges(self.extra_edge_indexs,self.extra_edge_types,1)

        return self.gcn.find_high_order_neigh(all_emb=self.all_embed,
                                               extra_edge_index=extra_edge_index,
                                               extra_edge_type=extra_edge_type,
                                               head_dict=head_dict,
                                               n_users=self.n_users,
                                               n_relations=self.n_relations,
                                               batch_size=batch_size,
                                               Ks=self.neighs)
