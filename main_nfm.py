import random
import itertools
from time import time

import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from model.NFM import NFM
from parser.parser_nfm import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_nfm import DataLoaderNFM


def evaluate_batch(model, dataloader, user_ids, K):
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    n_users = len(user_ids)
    n_items = dataloader.n_items
    item_ids = list(range(n_items))
    user_idx_map = dict(zip(user_ids, range(n_users)))

    feature_values = dataloader.generate_test_batch(user_ids)
    with torch.no_grad():
        scores = model(feature_values, is_train=False)              # (batch_size)

    rows = [user_idx_map[u] for u in np.repeat(user_ids, n_items).tolist()]
    cols = item_ids * n_users
    score_matrix = torch.Tensor(sp.coo_matrix((scores, (rows, cols)), shape=(n_users, n_items)).todense())

    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    precision_k, recall_k, ndcg_k, ndcg_truncate_k = calc_metrics_at_k(score_matrix, train_user_dict, test_user_dict, user_ids, item_ids, K)

    score_matrix = score_matrix.numpy()
    return score_matrix, precision_k, recall_k, ndcg_k, ndcg_truncate_k


def evaluate_mp(model, dataloader, K, num_processes, device):
    test_batch_size = dataloader.test_batch_size
    test_user_dict = dataloader.test_user_dict

    model.eval()
    model.to("cpu")

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    pool = mp.Pool(num_processes)
    res = pool.starmap(evaluate_batch, [(model, dataloader, batch_user, K) for batch_user in user_ids_batches])
    pool.close()

    score_matrix    = np.concatenate([r[0] for r in res], axis=0)
    precision_k     = np.concatenate([r[1] for r in res])
    recall_k        = np.concatenate([r[2] for r in res])
    ndcg_k          = np.concatenate([r[3] for r in res])
    ndcg_truncate_k = np.concatenate([r[4] for r in res])

    precision_k = precision_k.mean()
    recall_k = recall_k.mean()
    ndcg_k = ndcg_k.mean()
    ndcg_truncate_k = ndcg_truncate_k.mean()

    torch.cuda.empty_cache()
    model.to(device)
    return score_matrix, precision_k, recall_k, ndcg_k, ndcg_truncate_k


def evaluate(model, dataloader, K, num_processes, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]

    n_users = len(user_ids)
    n_items = dataloader.n_items
    item_ids = list(range(n_items))
    user_idx_map = dict(zip(user_ids, range(n_users)))

    cf_users = []
    cf_items = []
    cf_scores = []

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user in user_ids_batches:
            feature_values = dataloader.generate_test_batch(batch_user)
            feature_values = feature_values.to(device)

            with torch.no_grad():
                batch_scores = model(feature_values, is_train=False)            # (batch_size)

            cf_users.extend(np.repeat(batch_user, n_items).tolist())
            cf_items.extend(item_ids * len(batch_user))
            cf_scores.append(batch_scores.cpu())
            pbar.update(1)

    rows = [user_idx_map[u] for u in cf_users]
    cols = cf_items
    cf_scores = torch.cat(cf_scores)
    cf_score_matrix = torch.Tensor(sp.coo_matrix((cf_scores, (rows, cols)), shape=(n_users, n_items)).todense())

    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    precision_k, recall_k, ndcg_k, ndcg_truncate_k = calc_metrics_at_k(cf_score_matrix, train_user_dict, test_user_dict, user_ids, item_ids, K)

    cf_score_matrix = cf_score_matrix.numpy()
    precision_k = precision_k.mean()
    recall_k = recall_k.mean()
    ndcg_k = ndcg_k.mean()
    ndcg_truncate_k = ndcg_truncate_k.mean()
    return cf_score_matrix, precision_k, recall_k, ndcg_k, ndcg_truncate_k


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderNFM(args, logging)
    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # construct model & optimizer
    model = NFM(args, data.n_users, data.n_items, data.n_entities, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1
    epoch_list = []

    precision_list = []
    recall_list = []
    ndcg_list = []
    ndcg_truncate_list = []

    num_processes = args.test_cores
    if num_processes and num_processes > 1:
        evaluate_func = evaluate_mp
    else:
        evaluate_func = evaluate

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
            pos_feature_values, neg_feature_values = data.generate_train_batch(data.train_user_dict)
            pos_feature_values = pos_feature_values.to(device)
            neg_feature_values = neg_feature_values.to(device)
            batch_loss = model([pos_feature_values, neg_feature_values], is_train=True)

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
            _, precision, recall, ndcg, ndcg_truncate = evaluate_func(model, data, args.K, num_processes, device)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f} Truncated NDCG {:.4f}'.format(epoch, time() - time1, precision, recall, ndcg, ndcg_truncate))

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
    _, precision, recall, ndcg, ndcg_truncate = evaluate_func(model, data, args.K, num_processes, device)
    logging.info('Final CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}  Truncated NDCG {:.4f}'.format(precision, recall, ndcg, ndcg_truncate))

    epoch_list.append(epoch)
    precision_list.append(precision)
    recall_list.append(recall)
    ndcg_list.append(ndcg)
    ndcg_truncate_list.append(ndcg_truncate)

    metrics = pd.DataFrame([epoch_list, precision_list, recall_list, ndcg_list, ndcg_truncate_list]).transpose()
    metrics.columns = ['epoch_idx', 'precision@{}'.format(args.K), 'recall@{}'.format(args.K), 'ndcg@{}'.format(args.K), 'ndcg_truncate@{}'.format(args.K)]
    metrics.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)


def predict(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data = DataLoaderNFM(args, logging)

    # load model
    model = NFM(args, data.n_users, data.n_items, data.n_entities)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    num_processes = args.test_cores
    if num_processes and num_processes > 1:
        evaluate_func = evaluate_mp
    else:
        evaluate_func = evaluate

    cf_scores, precision, recall, ndcg, ndcg_truncate = evaluate_func(model, data, args.K, num_processes, device)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}  Truncated NDCG {:.4f}'.format(precision, recall, ndcg, ndcg_truncate))



if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    args = parse_nfm_args()
    train(args)
    # predict(args)


