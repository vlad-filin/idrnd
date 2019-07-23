import numpy as np
import librosa
from scipy.io import wavfile
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
from pathlib import Path
import os
from kekas.callbacks import Callback
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.functional import softmax

def step_fn(model: torch.nn.Module,
            batch: torch.Tensor) -> torch.Tensor:
    """Determine what your model will do with your data.

    Args:
        model: the pytorch module to pass input in
        batch: the batch of data from the DataLoader

    Returns:
        The models forward pass results
    """

    voice = batch['data']
    if 'mfcc' in batch.keys():
        return model(voice, batch['mfcc'])
    return model(voice)


class ScoreCallback(Callback):
    def __init__(self, preds_key, target_key, metric_fn, path_to_save='eer_checkpoints',
                 logdir="tensorboard"):

        self.preds_key = preds_key
        self.target_key = target_key
        self.human_probs_for_human = []
        self.human_probs_for_spoof = []
        self.spoof_probs = []
        self.targets = []
        self.softmax = nn.Softmax(dim=1)
        self.metric_fn = metric_fn
        self.top3_scores = []
        self.path = Path(path_to_save)
        self.logdir = Path(logdir)

    def on_train_begin(self, state) -> None:
        self.logdir.mkdir(exist_ok=True)
        self.path.mkdir(exist_ok=True)
        self.val_writer = SummaryWriter(str(self.logdir / "val_eer"))

    def on_batch_end(self, i, state) -> None:
        if state.core.mode == "val":

            preds = state.core.out[self.preds_key]
            if isinstance(preds, list):
                res = []
                for p in preds:
                    res.append(self.softmax(p).cpu().numpy())
                preds = np.concatenate(res, axis=0)
            else:
                preds = self.softmax(preds).cpu().numpy()
            targets = state.core.batch[self.target_key].cpu().numpy()
            tmp = preds[targets == 0, 0]
            if len(tmp) > 0:
                self.human_probs_for_human.append(tmp)
            tmp = preds[targets == 1, 0]
            if len(tmp) > 0:
                self.human_probs_for_spoof.append(tmp)
            self.spoof_probs.append(preds[:, 1, ...])
            self.targets.append(targets)

    def on_epoch_end(self, epoch, state) -> None:
        if state.core.mode == "val":
            print("_" * 50)
            print("Metric computation")
            self.human_probs_for_human = np.hstack(self.human_probs_for_human)
            self.human_probs_for_spoof = np.hstack(self.human_probs_for_spoof)
            self.spoof_probs = np.hstack(self.spoof_probs)
            self.targets = np.hstack(self.targets)

            auc = roc_auc(self.targets, self.spoof_probs)
            eer, threshold = self.metric_fn(self.human_probs_for_human, self.human_probs_for_spoof)
            self.human_probs_for_human = []
            self.human_probs_for_spoof = []
            self.spoof_probs = []
            self.targets = []
            print('score 1 (eer): {0:.7f} threshold (?): {1:.4f} auc:  {2:.7f}' .format(
                eer, threshold, auc))
            self.val_writer.add_scalar("eer", eer, global_step=epoch)
            name = str(round(eer, 4)) + "." + str(epoch) + '.pt'
            name = os.path.join(self.path, name)
            if len(self.top3_scores) < 25:
                torch.save(state.core.model.state_dict(), name)
                self.top3_scores.append((round(eer, 4), name))
                self.top3_scores = sorted(self.top3_scores, key=lambda item: item[0])
            else:
                if eer < self.top3_scores[-1][0]:
                    os.remove(self.top3_scores[-1][1])
                    self.top3_scores.pop(-1)
                    torch.save(state.core.model.state_dict(), name)
                    self.top3_scores.append((round(eer, 4), name))
                    self.top3_scores = sorted(self.top3_scores, key=lambda item: item[0])


def exp_decay(epoch, k=0.1, initial_rate=0.0001):
    return initial_rate * np.exp(-k * epoch)


def jointer(abs_path, rel_path):
    with open(rel_path, 'r') as f:
        list_rel_path = f.readlines()
    res_path = [os.path.join(abs_path, rp.strip("\n")) for rp in list_rel_path]
    return res_path

def roc_auc(target, preds):
    target = target.astype(np.int)
    return roc_auc_score(target, preds)