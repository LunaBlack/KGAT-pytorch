import random
import logging
import argparse
from time import time

import torch
import numpy as np
import pandas as pd
import torch.optim as optim

from model.KGAT import KGAT
from utility.parser import *
from utility.log_helper import *
from utility.metrics import *
from utility.helper import *
from utility.load_data import DataLoader


def evaluate(model, train_graph, train_user_dict, test_user_dict, item_ids, K, use_cuda, device):
    model.eval()

    with torch.no_grad():
        att = model.compute_attention(train_graph)
    train_graph.edata['att'] = att

    user_ids = test_user_dict.keys()
    user_ids = torch.LongTensor(list(user_ids))
    if use_cuda:
        user_ids = user_ids.to(device)

    cf_scores = model.cf_score(train_graph, user_ids, item_ids)       # (n_eval_users, n_eval_items)
    precision_k, recall_k, ndcg_k = calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, K, use_cuda)
    return precision_k, recall_k, ndcg_k


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
    data = DataLoader(args, logging)
    if args.use_pretrain:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, user_pre_embed, item_pre_embed)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # move graph data to GPU
    train_graph = data.train_graph
    train_nodes = torch.LongTensor(train_graph.ndata['id'])
    train_edges = torch.LongTensor(train_graph.edata['type'])
    if use_cuda:
        train_nodes = train_nodes.to(device)
        train_edges = train_edges.to(device)
    train_graph.ndata['id'] = train_nodes
    train_graph.edata['type'] = train_edges

    test_graph = data.test_graph
    test_nodes = torch.LongTensor(test_graph.ndata['id'])
    test_edges = torch.LongTensor(test_graph.edata['type'])
    if use_cuda:
        test_nodes = test_nodes.to(device)
        test_edges = test_edges.to(device)
    test_graph.ndata['id'] = test_nodes
    test_graph.edata['type'] = test_edges

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # initialize metrics
    precision_list = []
    recall_list = []
    ndcg_list = []

    # train model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # update attention scores
        with torch.no_grad():
            att = model.compute_attention(train_graph)
        train_graph.edata['att'] = att
        logging.info('Update attention scores: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict)
            if use_cuda:
                cf_batch_user = cf_batch_user.to(device)
                cf_batch_pos_item = cf_batch_pos_item.to(device)
                cf_batch_neg_item = cf_batch_neg_item.to(device)
            cf_batch_loss = model.cf_loss(train_graph, cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)

            cf_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time1 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time2 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.train_kg_dict)
            if use_cuda:
                kg_batch_head = kg_batch_head.to(device)
                kg_batch_relation = kg_batch_relation.to(device)
                kg_batch_pos_tail = kg_batch_pos_tail.to(device)
                kg_batch_neg_tail = kg_batch_neg_tail.to(device)
            kg_batch_loss = model.kg_embedding(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail)

            kg_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % args.kg_print_every) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time2, kg_batch_loss.item(), kg_total_loss / iter))
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time1, kg_total_loss / n_kg_batch))

        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0:
            time1 = time()
            precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict, item_ids, args.K, use_cuda, device)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(epoch, time() - time1, precision, recall, ndcg))

            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            best_recall, should_stop = early_stopping(recall_list, args.stopping_steps)

            if should_stop:
                break

            if best_recall == recall_list[-1]:
                save_model(model, args.save_dir, epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))

    # save model
    save_model(model, args.save_dir, epoch)

    precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict, item_ids, args.K, use_cuda, device)
    logging.info('Final CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))



if __name__ == '__main__':
    args = parse_args()
    train(args)






