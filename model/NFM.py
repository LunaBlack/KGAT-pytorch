import torch
from torch import nn
from torch.nn import functional as F


class HiddenLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout):
        super(HiddenLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU()
        self.message_dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = self.linear(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.message_dropout(out)
        return out


class NFM(nn.Module):

    def __init__(self, args,
                 n_users, n_items, n_entities,
                 user_pre_embed=None, item_pre_embed=None):

        super(NFM, self).__init__()
        self.model_type = args.model_type
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_features = n_entities + n_users

        self.embed_dim = args.embed_dim

        self.hidden_dim_list = [args.embed_dim] + eval(args.hidden_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.hidden_dim_list))

        self.linear = nn.Linear(self.n_features, 1)

        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed, gain=nn.init.calculate_gain('relu'))
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.feature_embed = nn.Parameter(entity_user_embed)
        else:
            self.feature_embed = nn.Parameter(torch.Tensor(self.n_features, self.embed_dim))
            nn.init.xavier_uniform_(self.feature_embed, gain=nn.init.calculate_gain('relu'))

        self.batch_norm = nn.BatchNorm1d(self.embed_dim)
        self.dropout = nn.Dropout(self.mess_dropout[0])

        if self.model_type == 'fm':
            self.h = nn.Linear(self.embed_dim, 1, bias=False)
            with torch.no_grad():
                self.h.weight.copy_(torch.ones([1, self.embed_dim]))
            for param in self.h.parameters():
                param.requires_grad = False

        elif self.model_type == 'nfm':
            self.hidden_layers = nn.ModuleList()
            for idx in range(self.n_layers):
                self.hidden_layers.append(HiddenLayer(self.hidden_dim_list[idx], self.hidden_dim_list[idx + 1], self.mess_dropout[idx + 1]))
            self.h = nn.Linear(self.hidden_dim_list[-1], 1, bias=False)


    def predict(self, feature_values):
        """
        feature_values:   (batch_size, n_features), n_features = n_entities + n_users, torch.sparse.FloatTensor
        """
        # Bi-Interaction layer
        # Equation (4) / (3)
        sum_square_embed = torch.sparse.mm(feature_values, self.feature_embed).pow(2)           # (batch_size, embed_dim)
        square_sum_embed = torch.sparse.mm(feature_values.pow(2), self.feature_embed.pow(2))    # (batch_size, embed_dim)
        bi = 0.5 * (sum_square_embed - square_sum_embed)                                        # (batch_size, embed_dim)

        # Hidden layers
        z = self.dropout(self.batch_norm(bi))               # (batch_size, embed_dim)

        if self.model_type == 'nfm':
            # Equation (5)
            for i, layer in enumerate(self.hidden_layers):
                z = layer(z)                                # (batch_size, hidden_dim)

        # Prediction layer
        # Equation (6)
        y = self.h(z)                                       # (batch_size, 1)
        # Equation (2) / (7) / (8)
        y = self.linear(feature_values) + y                 # (batch_size, 1)
        return y.squeeze()                                  # (batch_size)


    def calc_loss(self, pos_feature_value, neg_feature_value):
        """
        pos_feature_value:  (batch_size, n_features), torch.sparse.FloatTensor
        neg_feature_value:  (batch_size, n_features), torch.sparse.FloatTensor
        """
        pos_score = self.predict(pos_feature_value)         # (batch_size)
        neg_score = self.predict(neg_feature_value)         # (batch_size)

        loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        loss = torch.mean(loss)
        return loss





