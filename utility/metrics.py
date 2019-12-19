import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error


def calc_recall(rank, ground_truth, k):
    """
    calculate recall of one example
    """
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(hit, k):
    """
    calculate Precision@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return np.mean(hit)


def average_precision(hit, cut):
    """
    calculate average precision (area under PR curve)
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)
    precisions = [precision_at_k(hit, k + 1) for k in range(cut) if len(hit) >= k]
    if not precisions:
        return 0.
    return np.sum(precisions) / float(min(cut, np.sum(hit)))


def dcg_at_k(rel, k):
    """
    calculate discounted cumulative gain (dcg)
    rel: list, element is positive real values, can be binary
    """
    rel = np.asfarray(rel)[:k]
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return dcg


def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def calc_auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def logloss(ground_truth, prediction):
    logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
    return logloss


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, K, use_cuda):
    """
    cf_scores: (n_eval_users, n_eval_items)
    """
    if use_cuda:
        user_ids = user_ids.cpu().numpy()
        item_ids = item_ids.cpu().numpy()
        cf_scores = cf_scores.cpu()
    else:
        user_ids = user_ids.numpy()
        item_ids = item_ids.numpy()
        cf_scores = cf_scores

    precision_all = []
    recall_all = []
    ndcg_all = []

    for user_id, test_pos_item_list in test_user_dict.items():
        user_idx = np.where(user_ids == user_id)[0][0]
        train_pos_item_list = train_user_dict[user_id]

        user_scores = cf_scores[user_idx]
        for item_id in train_pos_item_list:
            user_scores[item_id] = 0

        _, rank_indices = torch.sort(user_scores, descending=True)
        binary_hit = np.zeros(len(item_ids), dtype=np.float32)
        for idx in range(len(item_ids)):
            if rank_indices[idx].item() in test_pos_item_list:
                binary_hit[idx] = 1

        precision = precision_at_k(binary_hit, K)
        recall = recall_at_k(binary_hit, K, len(test_pos_item_list))
        ndcg = ndcg_at_k(binary_hit, K)

        precision_all.append(precision)
        recall_all.append(recall)
        ndcg_all.append(ndcg)

    precision_mean = sum(precision_all) / len(precision_all)
    recall_mean = sum(recall_all) / len(recall_all)
    ndcg_mean = sum(ndcg_all) / len(ndcg_all)
    return precision_mean, recall_mean, ndcg_mean







