import random
import logging
import argparse
from time import time

import torch
import numpy as np
import pandas as pd
import torch.optim as optim

from model.BPRMF import BPRMF
from utility.parser_bprmf import *
from utility.log_helper import *
from utility.metrics import *
from utility.helper import *
from utility.loader_bprmf import DataLoaderBPRMF


def evaluate(model, train_user_dict, test_user_dict, user_ids, item_ids, K, use_cuda):
    model.eval()

    with torch.no_grad():
        cf_scores = model.predict(user_ids, item_ids)       # (n_eval_users, n_eval_items)
    precision_k, recall_k, ndcg_k = calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, K, use_cuda)
    return cf_scores, precision_k, recall_k, ndcg_k


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

    user_ids = data.test_user_dict.keys()
    user_ids = torch.LongTensor(list(user_ids))
    if use_cuda:
        user_ids = user_ids.to(device)

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # construct model & optimizer
    model = BPRMF(args, data.n_users, data.n_items, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    epoch_list = []
    precision_list = []
    recall_list = []
    ndcg_list = []

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
            batch_loss = model.calc_loss(batch_user, batch_pos_item, batch_neg_item)

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
            _, precision, recall, ndcg = evaluate(model, data.train_user_dict, data.test_user_dict, user_ids, item_ids, args.K, use_cuda)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(epoch, time() - time1, precision, recall, ndcg))

            epoch_list.append(epoch)
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
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
    _, precision, recall, ndcg = evaluate(model, data.train_user_dict, data.test_user_dict, user_ids, item_ids, args.K, use_cuda)
    logging.info('Final CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))

    epoch_list.append(epoch)
    precision_list.append(precision)
    recall_list.append(recall)
    ndcg_list.append(ndcg)

    metrics = pd.DataFrame([epoch_list, precision_list, recall_list, ndcg_list]).transpose()
    metrics.columns = ['epoch_idx', 'precision@{}'.format(args.K), 'recall@{}'.format(args.K), 'ndcg@{}'.format(args.K)]
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

    user_ids = data.test_user_dict.keys()
    user_ids = torch.LongTensor(list(user_ids))
    if use_cuda:
        user_ids = user_ids.to(device)

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # load model
    model = BPRMF(args, data.n_users, data.n_items)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # predict
    cf_scores, precision, recall, ndcg = evaluate(model, data.train_user_dict, data.test_user_dict, user_ids, item_ids, args.K, use_cuda)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores.cpu().numpy())
    print('CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))



if __name__ == '__main__':
    args = parse_bprmf_args()
    train(args)
    # predict(args)






