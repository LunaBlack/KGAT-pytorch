import os
import torch


def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_model(model, model_dir, epoch_idx):
    model_state_file = os.path.join(model_dir, 'model_epoch%d.pth'.format(epoch_idx))
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch_idx}, model_state_file)






