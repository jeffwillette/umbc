import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch
from sklearn.metrics import adjusted_rand_score  # type: ignore
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.distributions import Normal

from utils import ece_partial, ece_partial_final, reg_cal_err

__all__ = ["Stats"]


T = torch.Tensor


class Stats:
    """
    Stats is used to track stats during both training and validation.

    There are many common stat types in machine learning which are calculated
    in similar ways, so this is an abtract class to track any type of stat
    with a few values as kwargs. Once an 'update' function and a 'crunch'
    function are defined, everything else should just work as expected
    """
    y_sigma: float
    y_mu: float

    stat_attributes = {
        "correct": 0.0,
        "acc_total": 0.0,
        "loss": 0.0,
        "iou": torch.Tensor(),
        "iou_n": 0.0,
        "m_iou": torch.Tensor(),
        "mean": 0.0,
        "loss_total": 0.0,
        "ll": 0.0,
        "nll": 0.0,
        "nll_total": 0.0,
        "accs": 0.0,
        "confs": 0.0,
        "n_in_bins": 0.0,
        "n": 0.0,
        "adj_rand_idx": 0.0,
        "adj_rand_idx_total": 0.0,
        "aupr": 0.0,
        "auroc": 0.0,
        "softmax_entropy": 0.0,
        "softmax_entropy_total": 0.0,
        "y_true": [],
        "y_score": [],

        # stats for regression
        "mu": [],
        "sigma": [],
        "y": [],
        "mse": 0.0,
        "mse_total": 0.0,
        "y_mu": 0.0,
        "y_sigma": 0.0

    }

    logs = ["id_ood_entropy"]

    # stats are things which can be tracked throughout training and then
    # logged once at the end of training
    stats = [
        "m_iou", "mean", "accuracy", "loss", "nll",
        "ece", "aupr", "auroc", "reg_ece", "reg_nll",
        "mse", "softmax_entropy", "adj_rand_idx"
    ]

    def __init__(
        self,
        stats: List[str],
        logs: Optional[List[Tuple[str, str]]] = None
    ) -> None:
        for s in stats:
            if s not in self.stats:
                raise ValueError(f"stat: {s} needs to be one of: {self.stats}")

        self.stats_tracked = stats
        self.crunch_funcs = {
            "m_iou": self.crunch_iou,
            "accuracy": self.crunch_accuracy,
            "loss": self.crunch_loss,
            "nll": self.crunch_nll,
            "ece": self.crunch_ece,
            "aupr": self.crunch_aupr_auroc,
            "auroc": self.crunch_aupr_auroc,
            "mse": self.crunch_mse,
            "reg_ece": self.crunch_reg_ece,
            "reg_nll": self.crunch_reg_nll,
            "softmax_entropy": self.crunch_softmax_entropy,
            "adj_rand_idx": self.crunch_adj_rand_idx,
        }
        self.zero()

        # if the logs arleady exist for a previous run, we should overwrite the
        # file with a new blank file which has the current timestamp. Later
        # when we write wto the log with this class, we will append to the
        # file created here.
        self.logs_tracked = {}
        if logs is not None:
            for (log, file) in logs:
                if log not in self.logs:
                    raise ValueError(
                        f"{log=} is invalid (choices: {self.logs})")

                # create the directory path if it does not already exist
                path = os.path.split(file)[0]
                os.makedirs(path, exist_ok=True)

                with open(file, "w") as _:
                    pass

                # save the path under the log name so we can update
                # the lofgile later
                self.logs_tracked[log] = file

        self.crunched = False

    def zero(self) -> None:
        self.crunched = False

        for att in self.stat_attributes:
            if isinstance(self.stat_attributes[att], list):
                setattr(self, att, [])
                continue
            setattr(self, att, self.stat_attributes[att])

    def set(self, attrs: List[Tuple[str, Any]]) -> None:
        for (name, val) in attrs:
            setattr(self, name, val)

    def crunch(self) -> None:
        if not self.crunched:
            for stat_name in self.stats_tracked:
                # aupr/auroc might get crunched twice (no effect) if this stays
                # commented out, but this was the easiest way to handle this
                # for now.
                # if stat_name == "auroc" and self.auroc == 0.0:
                #     continue  # skip because this is included in aupr
                self.crunch_funcs[stat_name]()
            self.crunched = True

    def print(self) -> None:
        """print all the stats without logging them anywhere"""
        self.crunch()
        values = [getattr(self, v) for v in self.stats_tracked]
        names = [v for v in self.stats_tracked]

        for (n, v) in zip(names, values):
            print(f"{n}: {v:0.4f} ", end=" ")

    def get_stats(self) -> Dict[str, Any]:
        self.crunch()
        values = [getattr(self, v) for v in self.stats_tracked]
        names = [v for v in self.stats_tracked]

        out = {}
        for (k, v) in zip(names, values):
            out[k] = v
        return out

    def log_stats(self, path: str) -> Tuple[List[Any], ...]:
        self.crunch()

        if not os.path.exists(path):
            # make the directory path if it does not already exist
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "w+") as f:
                f.write(f"{','.join([v for v in self.stats_tracked])}\n")

        values = [getattr(self, v) for v in self.stats_tracked]
        names = [v for v in self.stats_tracked]
        with open(path, "a+") as f:
            f.write(f"{','.join([str(v) for v in values])}\n")

        self.zero()
        return names, values

    def log_stats_df(
        self,
        path: str,
        info_dict: Dict[str, Any]
    ) -> Tuple[List[Any], ...]:
        """
        this was made as an experimental new way to log stats using a dataframe
        instead of a manually created csv file
        """
        self.crunch()

        values = [getattr(self, v) for v in self.stats_tracked]
        values = [v.tolist() if isinstance(v, T) else v for v in values]

        names = [v for v in self.stats_tracked]
        # we are appending a single row, but all of the columns need to
        # be in a list
        data: Dict[str, Any] = {n: [v] for (n, v) in zip(names, values)}
        data["timestamp"] = str(datetime.now())
        for k in info_dict:
            data[k] = [info_dict[k]]

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(data)
        with open(path, 'a+') as f:
            # only write header if the current line is 0
            df.to_csv(f, mode='a', header=f.tell() == 0)

        # if os.path.exists(path):
        #     old_df = pd.read_csv(path)
        #     df = pd.concat(old_df, df)

        self.zero()
        return names, values

    def update_aupr_auroc(self, y_true: T, y_score: T) -> None:
        self.y_true.append(y_true.detach().cpu().long())  # type: ignore
        self.y_score.append(y_score.detach().cpu())  # type: ignore

    def crunch_aupr_auroc(self) -> None:
        y_true = torch.cat(self.y_true)  # type: ignore
        y_score = torch.cat(self.y_score)  # type: ignore

        # average precision score is only for the multiclass setting,
        # so only use this if the y_score has a larger class dimension.
        if len(y_score.size()) > 1 and y_score.size(1) > 1:
            y_one_hot = torch.zeros((y_true.size(0), y_score.size(1)))
            y_one_hot[torch.arange(y_one_hot.size(0)), y_true] = 1
            self.aupr = average_precision_score(y_one_hot, y_score)
            self.auroc = roc_auc_score(y_one_hot, y_score)
            return

        def optimal_thresh(fpr, tpr, thresholds, p=0):
            loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
            idx = np.argmin(loss, axis=0)
            return fpr[idx], tpr[idx], thresholds[idx]

        # taken from: originall worked for the multiclass setting
        # https://github.com/binli123/dsmil-wsi/blob/dbb5cab415fb4079f89d8c977c34efd533ee87fa/train_tcga.py#L111  # noqa
        fpr, tpr, threshold = roc_curve(y_true, y_score)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(
            fpr, tpr, threshold)  # TODO: get this function

        self.aupr = average_precision_score(y_true, y_score)
        self.auroc = roc_auc_score(y_true, y_score)
        self.optimal_thresh = threshold_optimal
        self.acc = ((y_score > self.optimal_thresh) == y_true).float().mean()

    def update_iou(self, iou_sum: T, iou_n: int) -> None:
        self.iou_n += iou_n  # type: ignore
        if self.iou.numel() == 0:  # type: ignore
            self.iou = iou_sum
            return

        self.iou += iou_sum  # type: ignore

    def crunch_iou(self) -> None:
        self.m_iou = self.iou / self.safe_denom(self.iou_n)  # type: ignore

    def update_acc(self, correct: int, n: int) -> None:
        self.correct += correct  # type: ignore
        self.acc_total += n  # type: ignore

    def crunch_accuracy(self) -> None:
        self.accuracy = self.correct / \
            self.safe_denom(self.acc_total)  # type: ignore

    def update_adj_rand_idx(self, yhat: T, y: T) -> None:
        self.adj_rand_idx += abs(adjusted_rand_score(y, yhat))  # type: ignore
        self.adj_rand_idx_total += 1  # type: ignore

    def crunch_adj_rand_idx(self) -> None:  # type: ignore
        self.adj_rand_idx = self.adj_rand_idx / \
            self.safe_denom(self.adj_rand_idx_total)  # type: ignore

    def update_softmax_entropy(self, logits: T, n: int, softmaxxed: bool = False) -> None:
        if not softmaxxed:
            logits = logits.softmax(dim=-1)

        logits = torch.clamp(logits, 1e-45)
        self.softmax_entropy += - \
            (logits * torch.log(logits)).sum(dim=-1).sum().item()  # type: ignore
        self.softmax_entropy_total += n  # type: ignore

    def crunch_softmax_entropy(self) -> None:
        self.softmax_entropy = self.softmax_entropy / \
            self.safe_denom(self.softmax_entropy_total)  # type: ignore

    def update_loss(self, loss: T, n: int) -> None:
        self.loss += loss.detach().cpu().item()  # type: ignore
        self.loss_total += n  # type: ignore

    def safe_denom(self, val: float) -> float:
        return val + 1e-10

    def crunch_loss(self) -> None:
        self.loss = self.loss / \
            self.safe_denom(self.loss_total)  # type: ignore

    def update_nll(self, logits: T, y: T, softmaxxed: bool = False) -> None:
        if not softmaxxed:
            logits = logits.softmax(dim=-1)

        logits = torch.clamp(logits, 1e-45).log()

        self.ll += torch.gather(logits.detach().cpu(), 1,
                                y.view(-1, 1).detach().cpu()).sum().item()  # type: ignore
        self.nll_total += y.size(0)  # type: ignore

    def log_id_ood_entropy(self, id_ood_label: T, logits: T, softmaxxed: bool = False) -> None:
        # there is nothing to crunch ofr this one since we just need to store a list of them and then log it later
        if not softmaxxed:
            logits = logits.softmax(dim=-1)

        logits = torch.clamp(logits, 1e-45)

        entropy = -(logits * torch.log(logits)).sum(dim=-1)
        with open(self.logs_tracked["id_ood_entropy"], "a+") as f:
            np.savetxt(f, torch.cat((id_ood_label.unsqueeze(-1).cpu(),
                       entropy.unsqueeze(-1).cpu()), dim=-1).numpy())

    def crunch_nll(self) -> None:
        # it is ok to take the log here because the value in update nll for classification is the softmax probability
        # which needs to be log(.)ed
        self.nll = -self.ll / self.safe_denom(self.nll_total)  # type: ignore

    def update_ece(self, logits: T, y: T, softmaxxed: bool = False) -> None:
        confs, accs, n_in_bins, n = ece_partial(
            y.detach().cpu(), logits.detach().cpu(), softmaxxed=softmaxxed)
        self.accs += accs  # type: ignore
        self.confs += confs  # type: ignore
        self.n_in_bins += n_in_bins  # type: ignore
        self.n += n  # type: ignore

    def crunch_ece(self) -> None:
        self.ece = ece_partial_final(
            self.confs, self.accs, self.n_in_bins, self.n)  # type: ignore

    def update_reg_ece(self, mu: T, sigma: T, y: T) -> None:
        self.mu.append(mu)  # type: ignore
        self.sigma.append(sigma)  # type: ignore
        self.y.append(y)  # type: ignore

    def crunch_reg_ece(self) -> None:
        self.reg_ece = reg_cal_err(torch.cat(self.mu), torch.cat(
            self.sigma), torch.cat(self.y)).item()  # type: ignore

    def update_mse(self, mu: T, y: T) -> None:
        if not all([self.y_mu, self.y_sigma]):  # type: ignore
            raise ValueError(
                "reg nll needs y_sigma value. Call set() with a valid value")

        y, mu = y * self.y_sigma + self.y_mu, mu * self.y_sigma + self.y_mu
        self.mse += ((y - mu) ** 2).sum()  # type: ignore
        self.mse_total += y.numel()  # type: ignore

    def crunch_mse(self) -> None:
        self.mse = torch.sqrt(
            self.mse / self.safe_denom(self.mse_total)).item()  # type: ignore

    def update_reg_nll(self, mu: T, sigma: T, y: T) -> None:
        if self.y_sigma is None:  # type: ignore
            raise ValueError(
                "reg nll needs y_sigma value. Call set() with a valid value")

        self.ll += Normal(mu, sigma).log_prob(y).sum()  # type: ignore
        self.nll_total += y.size(0)  # type: ignore

    def crunch_reg_nll(self) -> None:
        self.reg_nll = -(self.ll / self.safe_denom(self.nll_total)
                         ).item() + np.log(self.y_sigma)  # type: ignore
