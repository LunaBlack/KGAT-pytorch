import torch
import torch.nn as nn
import torch.nn.functional as F


class ECFKG(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations,
                 user_pre_embed=None, item_pre_embed=None):

        super(ECFKG, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = args.embed_dim

        self.relation_embed = nn.Embedding(self.n_relations, self.embed_dim)
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed, gain=nn.init.calculate_gain('relu'))
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)


    def predict(self, user_ids, item_ids, relation_id):
        """
        user_ids:       number of users to evaluate   (n_eval_users)
        item_ids:       number of items to evaluate   (n_eval_items)
        relation_id:    number of relations           (1)
        """
        r_embed = self.relation_embed(relation_id)          # (1, embed_dim)
        user_embed = self.entity_user_embed(user_ids)       # (n_eval_users, embed_dim)
        item_embed = self.entity_user_embed(item_ids)       # (n_eval_items, embed_dim)

        cf_score = torch.matmul((user_embed + r_embed), item_embed.transpose(0, 1))     # (n_eval_users, n_eval_items)
        cf_score = F.softmax(cf_score, dim=1)                                           # (n_eval_users, n_eval_items)
        return cf_score


    def calc_loss(self, h, r, pos_t, neg_t):
        """
        h:      (batch_size)
        r:      (batch_size)
        pos_t:  (batch_size)
        neg_t:  (batch_size)
        """
        r_embed = self.relation_embed(r)                 # (batch_size, embed_dim)
        h_embed = self.entity_user_embed(h)              # (batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)      # (batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)      # (batch_size, embed_dim)

        # Equation (1) + (5)
        pos_score = F.logsigmoid(torch.sum((h_embed + r_embed) * pos_t_embed, dim=1, keepdim=True))      # (batch_size, 1)
        neg_score = F.logsigmoid(torch.sum((h_embed + r_embed) * neg_t_embed, dim=1, keepdim=True))      # (batch_size, 1)

        kg_loss = neg_score - pos_score
        kg_loss = torch.mean(kg_loss)
        return kg_loss



