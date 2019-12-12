import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.conv import SAGEConv
import math


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)       # W in formula (6)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)   # W in formula (7)
        elif self.aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in formula (8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in formula (8)
        else:
            raise NotImplementedError


class KGAT(nn.Module):

    def __init__(self,
                 n_users,
                 n_entities, entity_dim,
                 n_relations, relation_dim,
                 weight_dim_list, mess_dropout, A_in, aggregation_type,
                 kg_l2loss_lambda, cf_l2loss_lambda):

        super(KGAT, self).__init__()

        self.A_in = A_in

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.weight_dim_list = [entity_dim] + weight_dim_list
        self.n_layers = len(weight_dim_list) - 1

        self.kg_l2loss_lambda = kg_l2loss_lambda
        self.cf_l2loss_lambda = cf_l2loss_lambda

        self.user_embed = nn.Embedding(n_users, entity_dim)
        self.entity_embed = nn.Embedding(n_entities, entity_dim)
        self.relation_embed = nn.Embedding(n_relations, relation_dim)

        self.W_r = nn.Parameter(torch.Tensor(n_relations, entity_dim, relation_dim))
        nn.init.xavier_uniform(self.W_r, gain=nn.init.calculate_gain('relu'))

        self.attention_weights = {}
        for k in range(1, self.n_layers + 1):
            if aggregation_type == 'gcn':
                self.attention_weights['W_gcn_%d' % k] = nn.Linear(self.weight_dim_list[k - 1], self.weight_dim_list[k])               # W in formula (6)
            elif aggregation_type == 'graphsage':
                self.attention_weights['W_graphsage_%d' % k] = nn.Linear(self.weight_dim_list[k - 1] * 2, self.weight_dim_list[k])     # W in formula (7)
            elif self.aggregation_type == 'bi-interaction':
                self.attention_weights['W_bi1_%d' % k] = nn.Linear(self.weight_dim_list[k - 1], self.weight_dim_list[k])               # W1 in formula (8)
                self.attention_weights['W_bi2_%d' % k] = nn.Linear(self.weight_dim_list[k - 1], self.weight_dim_list[k])               # W2 in formula (8)

        self.mess_drops = {}
        for k in range(1, self.n_layers + 1):
            self.mess_drops[k] = nn.Dropout(mess_dropout[k - 1])


    def create_gcn_embed(self):
        # formula (6) & (9)
        A = self.A_in                                                           # (n_users + n_entities, n_users + n_entities)
        ego_embed = torch.cat([self.user_embed, self.entity_embed], dim=0)      # (n_users + n_entities, entity_dim)
        all_embed = [ego_embed]

        for k in range(1, self.n_layers + 1):
            side_embed = torch.matmul(A, ego_embed)                                                                     # (n_users + n_entities, conv_dim)
            ego_embed = F.leaky_relu(torch.matmul(ego_embed + side_embed, self.attention_weights['W_gcn_%d' % k]))      # (n_users + n_entities, next_conv_dim)
            ego_embed = self.mess_drops[k](ego_embed)

            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        all_embed = torch.cat(all_embed, dim=1)
        user_embed, entity_embed = torch.split(all_embed, [self.n_users, self.n_entities], 0)
        return user_embed, entity_embed


    def create_graphsage_embed(self):
        # formula (7) & (9)
        A = self.A_in                                                           # (n_users + n_entities, n_users + n_entities)
        ego_embed = torch.cat([self.user_embed, self.entity_embed], dim=0)      # (n_users + n_entities, entity_dim)
        all_embed = [ego_embed]

        for k in range(1, self.n_layers + 1):
            side_embed = torch.matmul(A, ego_embed)                                                                     # (n_users + n_entities, conv_dim)
            cat_embed = torch.cat([ego_embed, side_embed], dim=1)                                                       # (n_users + n_entities, conv_dim * 2)

            ego_embed = F.leaky_relu(torch.matmul(cat_embed, self.attention_weights['W_graphsage_%d' % k]))             # (n_users + n_entities, next_conv_dim)
            ego_embed = self.mess_drops[k](ego_embed)

            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        all_embed = torch.cat(all_embed, dim=1)
        user_embed, entity_embed = torch.split(all_embed, [self.n_users, self.n_entities], 0)
        return user_embed, entity_embed


    def create_bi_interaction_embed(self):
        # formula (8) & (9)
        A = self.A_in                                                           # (n_users + n_entities, n_users + n_entities)
        ego_embed = torch.cat([self.user_embed, self.entity_embed], dim=0)      # (n_users + n_entities, entity_dim)
        all_embed = [ego_embed]

        for k in range(1, self.n_layers + 1):
            side_embed = torch.matmul(A, ego_embed)                                                                     # (n_users + n_entities, conv_dim)
            sum_embed = F.leaky_relu(torch.matmul(ego_embed + side_embed, self.attention_weights['W_bi1_%d' % k]))      # (n_users + n_entities, next_conv_dim)
            bi_embed = F.leaky_relu(torch.matmul(ego_embed * side_embed, self.attention_weights['W_bi2_%d' % k]))       # (n_users + n_entities, next_conv_dim)

            ego_embed = sum_embed + bi_embed
            ego_embed = self.mess_drops[k](ego_embed)

            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # formula (11)
        all_embed = torch.cat(all_embed, dim=1)
        user_embed, entity_embed = torch.split(all_embed, [self.n_users, self.n_entities], 0)
        return user_embed, entity_embed


    def build_model_phase_I(self):
        if self.aggregation_type == 'gcn':
            self.ua_embeddings, self.ea_embeddings = self.create_gcn_embed()

        elif self.aggregation_type == 'graphsage':
            self.ua_embeddings, self.ea_embeddings = self.create_graphsage_embed()

        elif self.aggregation_type == 'bi-interaction':
            self.ua_embeddings, self.ea_embeddings = self.create_bi_interaction_embed()

        self.u_e = self.ua_embeddings[self.users]               # (batch_size, attention_concat_dim)
        self.pos_i_e = self.ea_embeddings[self.pos_items]       # (batch_size, attention_concat_dim)
        self.neg_i_e = self.ea_embeddings[self.neg_items]       # (batch_size, attention_concat_dim)


    def build_loss_phase_I(self):
        # formula (12)
        pos_score = torch.sum(self.u_e * self.pos_i_e, dim=1)
        neg_score = torch.sum(self.u_e * self.neg_i_e, dim=1)

        # formula (13)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(self.u_e) + _L2_loss_mean(self.pos_i_e) + _L2_loss_mean(self.neg_i_e)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss


    def kg_embedding(self, h, r, pos_t, neg_t):
        r_embed = self.relation_embed(r)            # (batch_size, relation_dim)
        r_trans_M = self.W_r.index_select(0, r)     # (batch_size, entity_dim, relation_dim)

        h_embed = self.entity_embed(h)              # (batch_size, entity_dim)
        pos_t_embed = self.entity_embed(pos_t)      # (batch_size, entity_dim)
        neg_t_embed = self.entity_embed(neg_t)      # (batch_size, entity_dim)

        h_e = torch.bmm(h_embed.unsqueeze(1), r_trans_M).squeeze(1)             # (batch_size, relation_dim)
        pos_t_e = torch.bmm(pos_t_embed.unsqueeze(1), r_trans_M).squeeze(1)     # (batch_size, relation_dim)
        neg_t_e = torch.bmm(neg_t_embed.unsqueeze(1), r_trans_M).squeeze(1)     # (batch_size, relation_dim)
        return h_e, pos_t_e, neg_t_e, r_embed


    def kg_loss(self, h_e, r_e, pos_t_e, neg_t_e):
        # formula (1)
        pos_score = torch.sum(torch.pow(h_e + r_e - pos_t_e, 2), dim=1, keepdim=True)     # (batch_size, 1)
        neg_score = torch.sum(torch.pow(h_e + r_e - neg_t_e, 2), dim=1, keepdim=True)     # (batch_size, 1)

        # formula (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_e) + _L2_loss_mean(r_e) + _L2_loss_mean(pos_t_e) + _L2_loss_mean(neg_t_e)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def attention_score(self, h_e, r_e, t_e):
        pass






