import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import random
import logging
import argparse
from time import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
dist.init_process_group(backend="nccl")

from model.BPRMF import BPRMF
from utility.parser_bprmf import *
from utility.log_helper import *
from utility.metrics import *
from utility.model_helper import *
from utility.loader_bprmf import DataLoaderBPRMF


def evaluate(model, train_user_dict, test_user_dict, user_ids_batches, item_ids, K):
    model.eval()
    model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

    n_users = len(test_user_dict.keys())
    item_ids_batch = item_ids.cpu().numpy()

    cf_scores = []
    precision = []
    recall = []
    ndcg = []
    ndcg_truncate = []

    with torch.no_grad():
        for user_ids_batch in user_ids_batches:
            cf_scores_batch = model.predict(user_ids_batch, item_ids)       # (n_batch_users, n_eval_items)

            cf_scores_batch = cf_scores_batch.cpu()
            user_ids_batch = user_ids_batch.cpu().numpy()
            precision_batch, recall_batch, ndcg_batch, ndcg_truncate_batch = calc_metrics_at_k(cf_scores_batch, train_user_dict, test_user_dict, user_ids_batch, item_ids_batch, K)

            cf_scores.append(cf_scores_batch.numpy())
            precision.append(precision_batch)
            recall.append(recall_batch)
            ndcg.append(ndcg_batch)
            ndcg_truncate.append(ndcg_truncate_batch)

    cf_scores = np.concatenate(cf_scores, axis=0)
    precision_k = sum(np.concatenate(precision)) / n_users
    recall_k = sum(np.concatenate(recall)) / n_users
    ndcg_k = sum(np.concatenate(ndcg)) / n_users
    ndcg_truncate_k = sum(np.concatenate(ndcg_truncate)) / n_users
    return cf_scores, precision_k, recall_k, ndcg_k, ndcg_truncate_k


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    data = DataLoaderBPRMF(args, logging)

    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    user_ids = list(data.test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + args.test_batch_size] for i in range(0, len(user_ids), args.test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    if use_cuda:
        user_ids_batches = [d.to(device) for d in user_ids_batches]

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # construct model & optimizer
    model = BPRMF(args, data.n_users, data.n_items, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    if n_gpu > 1:
        model = nn.parallel.DistributedDataParallel(model)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    epoch_list = []
    precision_list = []
    recall_list = []
    ndcg_list = []
    ndcg_truncate_list = []

    # train model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # train cf
        time1 = time()
        total_loss = 0
        n_batch = data.n_cf_train // data.train_batch_size + 1

        for iter in range(1, n_batch + 1):
            time2 = time()
            batch_user, batch_pos_item, batch_neg_item = data.generate_train_batch(data.train_user_dict)
            if use_cuda:
                batch_user = batch_user.to(device)
                batch_pos_item = batch_pos_item.to(device)
                batch_neg_item = batch_neg_item.to(device)
            batch_loss = model('train', batch_user, batch_pos_item, batch_neg_item).mean()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

            if (iter % args.print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_batch, time() - time2, batch_loss.item(), total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_batch, time() - time1, total_loss / n_batch))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0:
            time1 = time()
            _, precision, recall, ndcg, ndcg_truncate = evaluate(model, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(epoch, time() - time1, precision, recall, ndcg))

            epoch_list.append(epoch)
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            ndcg_truncate_list.append(ndcg_truncate)
            best_recall, should_stop = early_stopping(recall_list, args.stopping_steps)

            if should_stop:
                break

            if recall_list.index(best_recall) == len(recall_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save model
    save_model(model, args.save_dir, epoch)

    # save metrics
    _, precision, recall, ndcg, ndcg_truncate = evaluate(model, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K)
    logging.info('Final CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))

    epoch_list.append(epoch)
    precision_list.append(precision)
    recall_list.append(recall)
    ndcg_list.append(ndcg)
    ndcg_truncate_list.append(ndcg_truncate)

    metrics = pd.DataFrame([epoch_list, precision_list, recall_list, ndcg_list, ndcg_truncate_list]).transpose()
    metrics.columns = ['epoch_idx', 'precision@{}'.format(args.K), 'recall@{}'.format(args.K), 'ndcg@{}'.format(args.K), 'ndcg@{}_truncate'.format(args.K)]
    metrics.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)


def predict(args):
    # GPU / CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    data = DataLoaderBPRMF(args, logging)

    user_ids = list(data.test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + args.test_batch_size] for i in range(0, len(user_ids), args.test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    if use_cuda:
        user_ids_batches = [d.to(device) for d in user_ids_batches]

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # load model
    model = BPRMF(args, data.n_users, data.n_items)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    cf_scores, precision, recall, ndcg, ndcg_truncate = evaluate(model, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f} NDCG_Truncate {:.4f}'.format(precision, recall, ndcg, ndcg_truncate))



if __name__ == '__main__':
    args = parse_bprmf_args()
    train(args)
    # predict(args)






