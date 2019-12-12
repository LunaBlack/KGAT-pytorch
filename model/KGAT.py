import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.conv import SAGEConv
import math


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def bmm_maybe_select(A, B, index):
    """Slice submatrices of B by the given index and perform bmm.
    B is a 3D tensor of shape (N, D1, D2), which can be viewed as a stack of
    N matrices of shape (D1, D2). The input index is an integer vector of length M.
    A could be either:
    (1) a dense tensor of shape (M, D1),
    (2) an integer vector of length M.
    The result C is a 2D matrix of shape (M, D2)
    For case (1), C is computed by bmm:
    ::
        C[i, :] = matmul(A[i, :], B[index[i], :, :])
    For case (2), C is computed by index select:
    ::
        C[i, :] = B[index[i], A[i], :]
    Parameters
    ----------
    A : torch.Tensor
        lhs tensor
    B : torch.Tensor
        rhs tensor
    index : torch.Tensor
        index tensor
    Returns
    -------
    C : torch.Tensor
        return tensor
    """
    if A.dtype == torch.int64 and len(A.shape) == 1:
        # following is a faster version of B[index, A, :]
        B = B.view(-1, B.shape[2])
        flatidx = index * B.shape[1] + A
        return B.index_select(0, flatidx)
    else:
        BB = B.index_select(0, index)
        return torch.bmm(A.unsqueeze(1), BB).squeeze()


class KGATConv(nn.Module):
    def __init__(self, entity_in_feats, out_feats, dropout, res_type="Bi"):
        super(KGATConv, self).__init__()
        self.mess_drop = nn.Dropout(dropout)
        self._res_type = res_type
        if res_type == "Bi":
            #self.res_fc = nn.Linear(entity_in_feats, out_feats, bias=False)
            self.res_fc_2 = nn.Linear(entity_in_feats, out_feats, bias=False)
        else:
            raise NotImplementedError

    def forward(self, g, nfeat):
        g = g.local_var()
        g.ndata['h'] = nfeat
        #g.ndata['h'] = torch.matmul(nfeat, self.W_r).squeeze() ### (#node, #rel, entity_dim)
        #print("relation_W", self.relation_W.shape,  self.relation_W)
        ### compute attention weight using edge_softmax
        #print("attention_score:", graph.edata['att_w'])
        #print("att_w", att_w)
        g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_neighbor'))
        h_neighbor = g.ndata['h_neighbor']
        if self._res_type == "Bi":
            # out = F.leaky_relu(self.res_fc(g.ndata['h']+h_neighbor))+\
            #       F.leaky_relu(self.res_fc_2(torch.mul(g.ndata['h'], h_neighbor)))
            out = F.leaky_relu(self.res_fc_2(torch.mul(g.ndata['h'], h_neighbor)))
        else:
            raise NotImplementedError

        return self.mess_drop(out)

class Model(nn.Module):
    def __init__(self, use_KG, input_node_dim, gnn_model, num_gnn_layers, n_hidden, dropout, use_attention=True,
                 n_entities=None, n_relations=None, relation_dim=None,
                 input_item_dim=None, input_user_dim=None, item_num=None, user_num=None,
                 use_pretrain=False, user_pre_embed=None, item_pre_embed=None,
                 reg_lambda_kg=0.01, reg_lambda_gnn=0.01, res_type="Bi"):
        super(Model, self).__init__()
        self._use_KG = use_KG
        self._n_entities = n_entities
        self._n_relations = n_relations
        self._gnn_model = gnn_model
        self._use_attention = use_attention
        self._use_pretrain = use_pretrain
        self._reg_lambda_kg = reg_lambda_kg
        self._reg_lambda_gnn = reg_lambda_gnn
        if use_pretrain:
            assert user_pre_embed is not None
            assert item_pre_embed is not None
            assert user_pre_embed.shape[1] == item_pre_embed.shape[1]

        ### for input node embedding
        if use_KG:
            self.entity_embed = nn.Embedding(n_entities, input_node_dim) ### e_h, e_t
            if use_pretrain:
                other_embed = nn.Parameter(torch.Tensor(n_entities-user_pre_embed.shape[0]-item_pre_embed.shape[0],
                                                     input_node_dim))
                nn.init.xavier_uniform_(other_embed, gain=nn.init.calculate_gain('relu'))
                self.entity_embed.weight = nn.Parameter(torch.cat((item_pre_embed, other_embed, user_pre_embed), dim=0))
            self.relation_embed = nn.Embedding(n_relations, relation_dim)  ### e_r
            self.W_R = nn.Parameter(torch.Tensor(n_relations, input_node_dim, relation_dim))  ### W_r
            nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))
        else:

            if input_item_dim:
                self.item_proj = nn.Linear(input_item_dim, input_node_dim, bias=False)
            else:
                self.item_proj = nn.Embedding(item_num, input_node_dim)

            if input_user_dim:
                self.user_proj = nn.Linear(input_user_dim, input_node_dim, bias=False)
            else:
                self.user_proj = nn.Embedding(user_num, input_node_dim)

            if use_pretrain:
                self.item_user_embed = nn.Embedding(item_num+user_num, user_pre_embed.shape[1])
                self.item_user_embed.weight = nn.Parameter(torch.cat((item_pre_embed, user_pre_embed), dim=0))

        self.layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            r = int(matorch.pow(2, i))
            act = None if i+1 == num_gnn_layers else F.relu
            if i==0:
                if (not use_KG) and use_pretrain:
                    in_dim = input_node_dim + item_pre_embed.shape[1]
                else:
                    in_dim = input_node_dim
                if gnn_model == "kgat":
                    self.layers.append(KGATConv(in_dim, n_hidden // r, dropout))
                elif gnn_model == "graphsage":
                    self.layers.append(SAGEConv(in_dim, n_hidden // r, aggregator_type="mean",
                                                feat_drop=dropout, activation=act))
                else:
                    raise NotImplementedError
            else:
                r2 = int(matorch.pow(2, i - 1))
                if gnn_model == "kgat":
                    self.layers.append(KGATConv(n_hidden // r2, n_hidden // r, dropout))
                elif gnn_model == "graphsage":
                    self.layers.append(SAGEConv(n_hidden // r2, n_hidden // r, aggregator_type="mean",
                                                feat_drop=dropout, activation=act))
                else:
                    raise NotImplementedError


    def transR(self, h, r, pos_t, neg_t):
        h_embed = self.entity_embed(h)  ### Shape(batch_size, dim)
        r_embed = self.relation_embed(r)
        pos_t_embed = self.entity_embed(pos_t)
        neg_t_embed = self.entity_embed(neg_t)

        h_vec = F.normalize(bmm_maybe_select(h_embed, self.W_R, r), p=2, dim=1)
        r_vec = F.normalize(r_embed, p=2, dim=1)
        pos_t_vec = F.normalize(bmm_maybe_select(pos_t_embed, self.W_R, r), p=2, dim=1)
        neg_t_vec = F.normalize(bmm_maybe_select(neg_t_embed, self.W_R, r), p=2, dim=1)

        pos_score = torch.sum(torch.pow(h_vec + r_vec - pos_t_vec, 2), dim=1, keepdim=True)
        neg_score = torch.sum(torch.pow(h_vec + r_vec - neg_t_vec, 2), dim=1, keepdim=True)
        ### pairwise ranking loss
        l = (-1.0) * F.logsigmoid(neg_score-pos_score)
        l = torch.mean(l)
        ## tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keepdims=True)
        ### TODO to check whether to use raw embeddings or entities embeddings
        # reg_loss =_L2_loss_mean(self.relation_embed.weight) + _L2_loss_mean(self.entity_embed.weight) + \
        #           _L2_loss_mean(self.W_R)
        reg_loss = _L2_loss_mean(h_vec) + _L2_loss_mean(r_vec) + \
                   _L2_loss_mean(pos_t_vec) + _L2_loss_mean(neg_t_vec)
        #print("\tkg loss:", l, "reg loss:", reg_loss, "*", self._reg_lambda_kg)
        loss = l + self._reg_lambda_kg * reg_loss
        return loss

    def _att_score(self, edges):
        """
        att_score = (W_r h_t)^T tanh(W_r h_r + e_r)

        """
        t_r = torch.matmul(self.entity_embed(edges.src['id']), self.W_r) ### (edge_num, hidden_dim)
        #print("t_r", t_r.shape, t_r)
        h_r = torch.matmul(self.entity_embed(edges.dst['id']), self.W_r) ### (edge_num, hidden_dim)
        #print("h_r", h_r.shape, h_r)
        att_w = torch.bmm(t_r.unsqueeze(1),
                       torch.tanh(h_r + self.relation_embed(edges.data['type'])).unsqueeze(2)).squeeze(-1)
        #print("att_w", att_w.shape, att_w)
        return {'att_w': att_w}

    def compute_attention(self, g):
        ## compute attention weight and store it on edges
        g = g.local_var()
        #print("In compute_attention ...", g)
        for i in range(self._n_relations):
            e_idxs = g.filter_edges(lambda edges: edges.data['type'] == i)
            self.W_r = self.W_R[i]
            g.apply_edges(self._att_score, e_idxs)
        g.edata['w'] = edge_softmax(g, g.edata.pop('att_w'))
        return g.edata.pop('w')

    def gnn(self, g, x):
        g = g.local_var()
        #print("In gnn ...", g)
        if self._use_KG:
            ### if use_KG : x = g.ndata['id']
            ### else:       u_fea, v_fea
            h = self.entity_embed(g.ndata['id'])
        else:
            h = torch.cat((self.item_proj(x[0]), self.user_proj(x[1])), dim=0)
            if self._use_pretrain:
                h2 = self.item_user_embed(g.ndata['id'])
                h = torch.cat((h, h2), dim=1)
        # if self._use_attention:
        #     g  = self.compute_attention(g, node_ids, rel_ids)
        node_embed_cache = [h]
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            # print(i, "h", h.shape, h)
            out = F.normalize(h, p=2, dim=1)
            #print(i, "norm_h", out.shape, out)
            node_embed_cache.append(out)
        final_h = torch.cat(node_embed_cache, 1)
        #print("final_h", final_h.shape, final_h)
        return final_h

    def get_loss(self, embedding, src_ids, pos_dst_ids, neg_dst_ids):
        src_vec = embedding[src_ids]
        pos_dst_vec = embedding[pos_dst_ids]
        neg_dst_vec = embedding[neg_dst_ids]
        pos_score = torch.bmm(src_vec.unsqueeze(1), pos_dst_vec.unsqueeze(2)).squeeze()  ### (batch_size, )
        neg_score = torch.bmm(src_vec.unsqueeze(1), neg_dst_vec.unsqueeze(2)).squeeze()  ### (batch_size, )
        #print("pos_score", pos_score)
        #print("neg_score", neg_score)
        cf_loss = torch.mean(F.logsigmoid(pos_score - neg_score) ) * (-1.0)
        ### TODO to check whether to use the raw embeddings or entity embeddings
        # reg_loss = _L2_loss_mean(self.relation_embed.weight) + _L2_loss_mean(self.entity_embed.weight) +\
        #            _L2_loss_mean(self.W_R)
        reg_loss = _L2_loss_mean(src_vec) + _L2_loss_mean(pos_dst_vec) + _L2_loss_mean(neg_dst_vec)
        #print("\tcf loss:", cf_loss, "reg loss:", reg_loss, "*", self._reg_lambda_gnn)
        return cf_loss + self._reg_lambda_gnn * reg_loss

