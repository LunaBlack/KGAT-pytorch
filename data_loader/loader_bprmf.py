import torch
import numpy as np
import pandas as pd

from data_loader.loader_base import DataLoaderBase


class DataLoaderBPRMF(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.train_batch_size = args.train_batch_size
        self.print_info(logging)


    def print_info(self, logging):
        logging.info('n_users:    %d' % self.n_users)
        logging.info('n_items:    %d' % self.n_items)
        logging.info('n_cf_train: %d' % self.n_cf_train)
        logging.info('n_cf_test:  %d' % self.n_cf_test)


    def generate_train_batch(self, user_dict):
        batch_user, batch_pos_item, batch_neg_item = self.generate_cf_batch(user_dict, self.train_batch_size)
        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item


