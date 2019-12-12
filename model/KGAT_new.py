import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax


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
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in formula (8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in formula (8)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()


    def forwad(self, g, entity_embed):
        g = g.local_var()
        g.ndata['node'] = entity_embed
        # formula (3) & (10)
        g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))

        if self.aggregator_type == 'gcn':
            # formula (6) & (9)
            out = self.activation(self.W(g.ndata['node'] + g.ndata['N_h']))                         # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'graphsage':
            # formula (7) & (9)
            out = self.activation(self.W(torch.cat([g.ndata['node'], g.ndata['N_h']], dim=1)))      # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'bi-interaction':
            # formula (8) & (9)
            out1 = self.activation(self.W1(g.ndata['node'] + g.ndata['N_h']))                       # (n_users + n_entities, out_dim)
            out2 = self.activation(self.W2(g.ndata['node'] * g.ndata['N_h']))                       # (n_users + n_entities, out_dim)
            out = out1 + out2
        else:
            raise NotImplementedError

        out = self.message_dropout(out)
        return out


class KGAT(nn.Module):

    def __init__(self,
                 n_users,
                 n_entities, entity_dim,
                 n_relations, relation_dim,
                 conv_dim_list, mess_dropout, aggregation_type,
                 kg_l2loss_lambda, cf_l2loss_lambda):

        super(KGAT, self).__init__()

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.entity_dim = entity_dim
        self.relation_dim = relation_dim

        self.conv_dim_list = [entity_dim] + conv_dim_list
        self.mess_dropout = mess_dropout
        self.n_layers = len(conv_dim_list) - 1

        self.kg_l2loss_lambda = kg_l2loss_lambda
        self.cf_l2loss_lambda = cf_l2loss_lambda

        self.user_entity_embed = nn.Embedding(n_users + n_entities, entity_dim)
        self.relation_embed = nn.Embedding(n_relations, relation_dim)

        self.W_R = nn.Parameter(torch.Tensor(n_relations, entity_dim, relation_dim))
        nn.init.xavier_uniform(self.W_R, gain=nn.init.calculate_gain('relu'))

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], aggregation_type))


    def att_score(self, edges):
        # formula (4)
        r_mul_t = torch.matmul(self.user_entity_embed(edges.src['id']), self.W_r)                       # (n_edge, relation_dim)
        r_mul_h = torch.matmul(self.user_entity_embed(edges.dst['id']), self.W_r)                       # (n_edge, relation_dim)
        r_embed = self.relation_embed(edges.data['type'])                                               # (1, relation_dim)
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)   # (n_edge, 1)
        return {'att': att}


    def compute_attention(self, g):
        g = g.local_var()
        for i in range(self.n_relations):
            edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_r = self.W_R[i]
            g.apply_edges(self.att_score, edge_idxs)

        # formula (5)
        g.edata['att'] = edge_softmax(g, g.edata.pop('att'))
        return g.edata.pop('att')


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


    def ckg_embedding(self, h, r, pos_t, neg_t):
        """
        h:      (ckg_batch_size)
        r:      (ckg_batch_size)
        pos_t:  (ckg_batch_size)
        neg_t:  (ckg_batch_size)
        """
        r_embed = self.relation_embed(r)                 # (ckg_batch_size, relation_dim)
        W_r = self.W_R[r]                                # (ckg_batch_size, entity_dim, relation_dim)

        h_embed = self.user_entity_embed(h)              # (ckg_batch_size, entity_dim)
        pos_t_embed = self.user_entity_embed(pos_t)      # (ckg_batch_size, entity_dim)
        neg_t_embed = self.user_entity_embed(neg_t)      # (ckg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)             # (ckg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)     # (ckg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)     # (ckg_batch_size, relation_dim)

        # formula (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1, keepdim=True)     # (ckg_batch_size, 1)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1, keepdim=True)     # (ckg_batch_size, 1)

        # formula (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def attention_embedding(self, g, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (att_batch_size)
        item_pos_ids:   (att_batch_size)
        item_neg_ids:   (att_batch_size)
        """
        g = g.local_var()
        ego_embed = self.user_entity_embed(g.ndata['id'])
        all_embed = [ego_embed]

        for i, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(g, ego_embed)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # formula (11)
        all_embed = torch.cat(all_embed, dim=1)

        user_embed = all_embed[user_ids]            # (att_batch_size, attention_concat_dim)
        item_pos_embed = all_embed[item_pos_ids]    # (att_batch_size, attention_concat_dim)
        item_neg_embed = all_embed[item_neg_ids]    # (att_batch_size, attention_concat_dim)

        # formula (12)
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)       # (att_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)       # (att_batch_size)

        # formula (13)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss





