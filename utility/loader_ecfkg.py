import os
import random
import collections

import torch
import numpy as np
import pandas as pd


class DataLoaderECFKG(object):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir

        self.train_batch_size = args.train_batch_size

        data_dir = os.path.join(args.data_dir, args.data_name)
        train_file = os.path.join(data_dir, 'train.txt')
        test_file = os.path.join(data_dir, 'test.txt')
        kg_file = os.path.join(data_dir, "kg_final.txt")

        self.cf_train_data, self.train_user_dict = self.load_cf(train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(test_file)
        self.statistic_cf()

        kg_data = self.load_kg(kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)

        if self.use_pretrain == 1:
            self.load_pretrained_data()


    def load_cf(self, filename):
        user = []
        item = []
        user_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))

                for item_id in item_ids:
                    user.append(user_id)
                    item.append(item_id)
                user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict


    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])


    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data


    def construct_data(self, kg_data):
        # plus inverse kg data
        n_relations = max(kg_data['r']) + 1
        reverse_kg_data = kg_data.copy()
        reverse_kg_data = reverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        reverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, reverse_kg_data], axis=0, ignore_index=True, sort=False)

        # re-map user id
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        # add interactions to kg data
        self.relation_u2i_id = 0

        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        reverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        reverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        reverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        cf2kg_test_data = pd.DataFrame(np.zeros((self.n_cf_test, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_test_data['h'] = self.cf_test_data[0]
        cf2kg_test_data['t'] = self.cf_test_data[1]

        reverse_cf2kg_test_data = pd.DataFrame(np.ones((self.n_cf_test, 3), dtype=np.int32), columns=['h', 'r', 't'])
        reverse_cf2kg_test_data['h'] = self.cf_test_data[1]
        reverse_cf2kg_test_data['t'] = self.cf_test_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, reverse_cf2kg_train_data], ignore_index=True)
        self.kg_test_data = pd.concat([kg_data, cf2kg_test_data, reverse_cf2kg_test_data], ignore_index=True)

        self.n_kg_train = len(self.kg_train_data)
        self.n_kg_test = len(self.kg_test_data)

        # construct kg dict
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)
        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.test_kg_dict = collections.defaultdict(list)
        self.test_relation_dict = collections.defaultdict(list)
        for row in self.kg_test_data.iterrows():
            h, r, t = row[1]
            self.test_kg_dict[h].append((t, r))
            self.test_relation_dict[r].append((h, t))


    def print_info(self, logging):
        logging.info('n_users:            %d' % self.n_users)
        logging.info('n_items:            %d' % self.n_items)
        logging.info('n_entities:         %d' % self.n_entities)
        logging.info('n_users_entities:   %d' % self.n_users_entities)
        logging.info('n_relations:        %d' % self.n_relations)

        logging.info('n_cf_train:         %d' % self.n_cf_train)
        logging.info('n_cf_test:          %d' % self.n_cf_test)

        logging.info('n_kg_train:         %d' % self.n_kg_train)
        logging.info('n_kg_test:          %d' % self.n_kg_test)


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=self.n_users_entities, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict):
        exist_heads = kg_dict.keys()
        if self.train_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.train_batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(self.train_batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data['user_embed']
        self.item_pre_embed = pretrain_data['item_embed']

        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.args.embed_dim
        assert self.item_pre_embed.shape[1] == self.args.embed_dim







