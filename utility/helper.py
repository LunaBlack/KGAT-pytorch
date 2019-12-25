import os
import dgl
import torch
from dgl import function as fn


def edge_softmax_fix(graph, score):
    # to solve the issue on `fn.sum`
    # it seems to get different results with `fn.sum` every time

    def reduce_sum(nodes):
        accum = torch.sum(nodes.mailbox['temp'], 1)
        return {'out_sum': accum}

    graph = graph.local_var()
    graph.edata['out'] = score
    graph.edata['out'] = torch.exp(graph.edata['out'])
    graph.update_all(fn.copy_e('out', 'temp'), reduce_sum)
    graph.apply_edges(fn.e_div_v('out', 'out_sum', 'out'))
    out = graph.edata['out']
    return out


def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model



