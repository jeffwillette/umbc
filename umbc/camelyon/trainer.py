import os
import signal
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import kornia.augmentation as K
import numpy as np
import pandas as pd  # type: ignore
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from base import Algorithm, HashableModule
from data.camelyon import CAMELYON16Patches256Extracted, CAMELYON16Patches256Iterable
from data.get import list_collate, pad_collate
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import roc_auc_score, roc_curve
from timm.utils import ModelEma
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm  # type: ignore
from transformers.optimization import get_linear_schedule_with_warmup
from utils import md5, seed, set_logger, str2bool

from umbc.models.camelyon import (
    CamelyonABMIL,
    CamelyonDeepSets,
    CamelyonDSMIL,
    CamelyonSSE,
    CamelyonSSEUMBC,
    IdentityExtractor,
    get_extractor,
)

T = torch.Tensor
A = NDArray
Model = Union[CamelyonDeepSets, CamelyonSSE]


class CustomAug(nn.Module):
    def __init__(self, img_size=256) -> None:
        super().__init__()
        rnd_hflip = K.RandomHorizontalFlip(p=0.5, same_on_batch=False)

        rnd_color_jitter = K.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
            p=0.8,
            same_on_batch=False,
        )

        kernel_size = int(img_size * 0.06)
        rnd_gaussian_blur = K.RandomGaussianBlur(
            kernel_size=(kernel_size, kernel_size),
            sigma=(0.1, 2.0),
            p=0.5,
            same_on_batch=False,
        )

        self.transforms = nn.Sequential(
            rnd_color_jitter,
            rnd_gaussian_blur,
            rnd_hflip,
        )

    @torch.no_grad()
    def forward(self, x):
        return self.transforms(x)


def optimal_thresh(
    fpr: NDArray, tpr: NDArray, thresholds: NDArray, p: float = 0
) -> Tuple[float, float, float]:
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


# taken from: originall worked for the multiclass setting
# https://github.com/binli123/dsmil-wsi/blob/dbb5cab415fb4079f89d8c977c34efd533ee87fa/train_tcga.py#L111  # noqa

# original used 80/20 random split between the 399 instances rather than
# the given test split:
# https://github.com/binli123/dsmil-wsi/issues/29
# https://github.com/binli123/dsmil-wsi/issues/22


def calc(y_true: NDArray, y_score: NDArray, thresh: float = 0.0) -> Tuple[float, ...]:
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    fpr_opt, tpr_opt, thresh_opt = optimal_thresh(fpr, tpr, threshold)

    if thresh > 0.0:
        thresh_opt = thresh

    aupr = average_precision_score(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    acc = ((y_score > thresh_opt) == y_true).mean()

    return aupr, auroc, acc, thresh_opt


class CamelyonTrainer(Algorithm):
    def __init__(self, args: Namespace, model: Model):
        super().__init__()

        self.args = args
        self.model: Union[Model, DDP] = model

        self.trainset: Dataset
        self.valset: Dataset

        self.best_score = 0.0
        self.epoch = 0
        self.finished = False
        self.loaded = False
        self.tuned = False

        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR

        self.train_ldr: DataLoader
        self.val_ldr: DataLoader

        self.train_sampler: DistributedSampler
        self.val_sampler: DistributedSampler

        self.bce_logits = F.binary_cross_entropy_with_logits
        self.rank: int
        self.model_ema = None

        addendum = ""
        if self.args.patch_dropout > 0.0:
            addendum = f"-patch-drop-{self.args.patch_dropout}"

        elif self.args.linear:
            self.log("linear scheduler")
            addendum = f"-linear-scheduler"
        if self.args.augmentation:
            addendum += "-augmentation"
        comment = f"-{args.comment}" if args.comment != "" else ""
        self.results_path = os.path.join(
            f"results{addendum}",
            f"{self.args.dataset}",  # type: ignore
            f"{self.get_model().name}" + comment,
        )

        self.models_path = os.path.join(self.results_path, "models")
        for d in [self.results_path, self.models_path]:
            os.makedirs(d, exist_ok=True)

        # write the model string to file under the model hash so we will
        # always know which model created this hash

        path = os.path.join(self.models_path, f"{self.args.model_hash}.txt")
        if os.path.exists(path):
            self.log(f"model hash: {self.args.model_hash} exists")
            return

        with open(path, "w") as f:
            self.log(f"writing new model hash: {self.args.model_hash}")
            f.write(self.args.model_string)

    def is_ddp(self) -> bool:
        if isinstance(self.model, DDP):
            if self.rank != 0:
                raise ValueError(
                    f"is ddp can only be called on rank 0 model: {self.rank=}"
                )
            return True
        return False

    def finetune_setup(self, rank: int) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpus[rank])
        self.rank = rank
        self.args.logger = set_logger("INFO")

        setup(self.rank, len(self.args.gpus), self.args.run)
        torch.cuda.set_device(self.args.gpus[self.rank])
        self.model = self.model.cuda()

        self.steps_per_epoch = math.ceil(270 / len(self.args.gpus))
        print(f"the number of steps per epoch {self.steps_per_epoch}")
        if self.args.model_ema:
            self.model_ema = ModelEma(self.model, self.args.model_ema_decay)
            self.log("Using EMA with decay = %.8f" % self.args.model_ema_decay)

        # load model must be handled in the main function before setup
        # is called
        self.model = DDP(
            self.model.cuda(),
            find_unused_parameters=False,
            device_ids=[self.args.gpus[self.rank]],
        )

        no_decay = ["bias", "LayerNorm.weight"]

        def dont_decay(n: str) -> bool:
            return any(nd in n for nd in no_decay)

        wd_params, no_wd_params, dec_params = [], [], []
        for n, p in self.model.named_parameters():
            # the way the resnet 18 is unrolled into a sequential block, these are the
            # prefixed for the projection and the first resnet block
            # if ("features.0" not in n and "features.4" not in n and not dont_decay(n)):
            # if (not dont_decay(n)):
            if "decoder" not in n and not dont_decay(n):
                wd_params.append(p)
                continue

            if "decoder" in n and not dont_decay(n):
                dec_params.append(p)
                continue

            if dont_decay(n):
                no_wd_params.append(p)

        wd, lr = 0.01, 1e-5
        print(f"learning rate: {lr}")
        optimizer_grouped_parameters = [
            {"params": wd_params, "weight_decay": wd},
            # {'params': dec_params, 'weight_decay': wd},
            {"params": no_wd_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

        def warmup_func(step: int) -> float:
            return (step + 1) / (self.steps_per_epoch + 1)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            (self.args.finetune_epochs - 1) * self.steps_per_epoch,
            lr / 10,
        )

        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=warmup_func
        )

        # self.optimizer = torch.optim.AdamW(
        #     optimizer_grouped_parameters, lr=lr)

        if self.args.linear:
            t_total = self.steps_per_epoch * self.args.finetune_epochs
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=0, num_training_steps=t_total
            )

        def get_train_loaders() -> None:
            n_cpu = os.cpu_count()
            assert isinstance(n_cpu, int)
            workers = n_cpu // len(self.args.gpus)

            # NOTE: this is divided by 5 to allow for the different runs
            workers = workers // 5

            self.log(f"inner workers: {workers}")

            self.trainset = CAMELYON16Patches256Iterable(
                root=self.args.data_root,
                inner_workers=workers,
                batch_size=self.args.chunk_size,
                split="train",
                p=self.args.patch_dropout,
            )
            self.valset = CAMELYON16Patches256Iterable(
                root=self.args.data_root,
                inner_workers=workers,
                batch_size=self.args.chunk_size,
                split="test",
            )
            self.log(f"{len(self.trainset)=} {len(self.valset)=}")

            self.train_sampler = DistributedSampler(
                self.trainset,
                num_replicas=len(self.args.gpus),
                rank=rank,
                shuffle=True,
                seed=self.args.run,
            )

            self.val_sampler = DistributedSampler(
                self.valset, num_replicas=len(self.args.gpus), rank=rank, shuffle=False
            )

            self.train_ldr = DataLoader(
                dataset=self.trainset,
                batch_size=1,
                collate_fn=list_collate,
                num_workers=2,
                sampler=self.train_sampler,
            )

            self.val_ldr = DataLoader(
                dataset=self.valset,
                batch_size=1,
                collate_fn=list_collate,
                num_workers=2,
                sampler=self.val_sampler,
            )

        def get_test_loader() -> None:
            get_train_loaders()

        # onnly used in finetuning
        self.augmentation = None
        if self.args.augmentation:
            self.augmentation = CustomAug()

        print(f"augmentation: {self.augmentation}")

        if self.args.mode == "finetune":
            self.log("getting loaders")
            get_train_loaders()
            self.log("fitting")

            self.log(
                "loaded resnet, saving the model for the first time "
                + "at the start of finetuning. If you want to restart a"
                + "previous finetuning run, then this needs to be changed"
            )
            self.save_model(self.models_path, "finetune")

            self.finetune_fit()
        elif self.args.mode == "finetune-test":
            self.log("getting loaders")
            get_test_loader()

            self.log("testing")
            tavg_loss, ty_true, ty_score = self.finetune_test()
            taupr, tauroc, tacc, tthresh = calc(ty_true, ty_score)
            test_stats_dict = {
                "acc": tacc,
                "aupr": taupr,
                "auroc": tauroc,
                "loss": tavg_loss,
                "thresh": tthresh,
            }

            if self.rank == 0:
                self.log_stats(self.results_path, "finetune_test", test_stats_dict)
        else:
            raise ValueError(f"got unknown mode: {self.args.mode}")

    def call_finetune_schedulers(self) -> None:
        if self.args.linear:
            self.scheduler.step()
        else:
            if self.epoch < 1:
                self.warmup_scheduler.step()
                return

            self.scheduler.step()

    def finetune_fit(self) -> None:
        self.epoch = 0
        self.finished = False

        while self.epoch < self.args.finetune_epochs:
            if isinstance(self.model, DDP):
                self.train_sampler.set_epoch(self.epoch)

            self.log("calling finetune train")
            avg_loss, y_true, y_score = self.finetune_train()
            aupr, auroc, acc, thresh = calc(y_true, y_score)
            train_stats_dict = {
                "acc": acc,
                "aupr": aupr,
                "auroc": auroc,
                "loss": avg_loss,
                "thresh": thresh,
            }

            if isinstance(self.model, DDP):
                dist.barrier()

            if isinstance(self.model, DDP):
                self.val_sampler.set_epoch(self.epoch)

            self.log("calling finetune test")
            tavg_loss, ty_true, ty_score = self.finetune_test()
            taupr, tauroc, tacc, tthresh = calc(ty_true, ty_score)
            test_stats_dict = {
                "acc": tacc,
                "aupr": taupr,
                "auroc": tauroc,
                "loss": tavg_loss,
                "thresh": tthresh,
            }

            if self.rank == 0:
                self.log_stats(self.results_path, "finetune_train", train_stats_dict)

                self.log_stats(self.results_path, "finetune_val", test_stats_dict)

                score = (tacc + tauroc) / 2
                self.log(f"score: {score} best: {self.best_score}")
                if score > self.best_score:
                    self.best_score = score

                    self.log("saving model")
                    self.save_model(self.models_path, "finetune")

            self.epoch += 1

        if self.rank == 0:
            self.log("finished finetuning, saving...")
            # model should have been saved the first time at
            # finetune initialization

            model_path = os.path.join(
                self.models_path, f"{self.args.model_hash}-finetune.pt"
            )
            dct = torch.load(model_path)
            dct["finished"] = True
            torch.save(dct, model_path)

    def finetune_train(self) -> Tuple[float, A, A]:
        self.model.train()

        total_y_true, total_y_score = [], []  # type: ignore
        total_loss, total_n = 0.0, 0
        for i, (batch_x, batch_y, _) in enumerate(self.train_ldr):
            self.optimizer.zero_grad()

            y_true, y_score = [], []  # type: ignore
            loss_accum, n = 0.0, 0
            for j, (x_loader, y) in enumerate(zip(batch_x, batch_y)):
                y = y.unsqueeze(0)

                grad_size = min(self.args.grad_set_size, len(x_loader.dataset))
                set_size = len(x_loader.dataset)

                remove_ext_handles = self.model.module.extractor.register_grad_correct_hooks(  # type: ignore # noqa
                    grad_size=grad_size, set_size=set_size
                )
                remove_pool_handles = self.model.module.mbc_pooler.register_grad_correct_hooks(  # type: ignore # noqa
                    grad_size=grad_size, set_size=set_size
                )

                ins_pred, bag_pred = self.model(
                    x_loader,
                    grad_size=self.args.grad_set_size,
                    test=False,
                    augmentation=self.augmentation,
                )

                ins_pred = ins_pred.view(-1).amax().view(1)
                bag_pred = bag_pred.view(-1)

                loss_ins = self.bce_logits(ins_pred, y.float().cuda())
                loss_bag = self.bce_logits(bag_pred, y.float().cuda())
                loss = (loss_ins + loss_bag) / (2 * len(batch_x))
                loss.backward()

                remove_ext_handles()
                remove_pool_handles()

                with torch.no_grad():
                    pred = (torch.sigmoid(ins_pred) + torch.sigmoid(bag_pred)) / 2

                    loss_accum += loss.detach().cpu().item() * len(batch_x)
                    y_score.extend(pred.tolist())
                    y_true.extend(y.tolist())
                    n += 1

            self.optimizer.step()
            self.call_finetune_schedulers()
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            obj_list: List[Dict[str, Any]] = [
                dict() for _ in range(len(self.args.gpus))
            ]
            vals = {
                "loss_accum": loss_accum,
                "y_true": y_true,
                "y_score": y_score,
                "n": n,
            }
            torch.distributed.all_gather_object(obj_list, vals)

            y_true, y_score = [], []
            for o in obj_list:
                total_loss += o["loss_accum"]
                total_n += o["n"]
                total_y_true.extend(o["y_true"])
                total_y_score.extend(o["y_score"])

            two_labels = len(np.unique(np.array(total_y_true)).tolist()) == 2
            if two_labels:
                _, auroc, acc, _ = calc(np.array(total_y_true), np.array(total_y_score))

                if self.rank == 0:
                    self.log(
                        f"loss: {total_loss / total_n:.4f} "
                        f"acc: {acc:.4f} auc: {auroc:.4f} "
                        f"iter: {i}/{len(self.train_ldr)}"
                    )

        return (
            total_loss / (total_n if total_n != 0 else 1),
            np.array(total_y_true),
            np.array(total_y_score),
        )

    def finetune_test(self) -> Tuple[float, A, A]:  # type: ignore
        if self.model_ema is not None:
            self.log("use ema for finetune-test")
            model = self.model_ema.ema
        else:
            model = self.model
        model.eval()
        total_y_true, total_y_score, total_used_idx = [], [], []
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for i, (batch_x, batch_y, x_idx) in enumerate(self.val_ldr):
                y_true, y_score, used_idx = [], [], []  # type: ignore
                loss_accum, n = 0.0, 0

                for j, (x_loader, y, idx) in enumerate(zip(batch_x, batch_y, x_idx)):

                    ins_pred, bag_pred = model(
                        x_loader, grad_size=self.args.grad_set_size, test=True
                    )

                    ins_pred = ins_pred.view(-1).amax().view(1)
                    bag_pred = bag_pred.view(-1)
                    y = y.unsqueeze(0)

                    # out is size (1) and y is size ()
                    loss_ins = self.bce_logits(ins_pred, y.float().cuda())
                    loss_bag = self.bce_logits(bag_pred, y.float().cuda())
                    loss = ((loss_ins + loss_bag) / 2).cpu().item()

                    pred = (torch.sigmoid(ins_pred) + torch.sigmoid(bag_pred)) / 2

                    loss_accum += loss * len(batch_x)
                    y_score.extend(pred.tolist())
                    y_true.extend(y.tolist())
                    n += 1
                    used_idx.append(idx)

                obj_list: List[Dict[str, Any]] = [
                    dict() for _ in range(len(self.args.gpus))
                ]
                vals = {
                    "loss_accum": loss_accum,
                    "y_true": y_true,
                    "y_score": y_score,
                    "n": n,
                    "used_idx": idx,
                }
                torch.distributed.all_gather_object(obj_list, vals)

                for o in obj_list:
                    if o["used_idx"] not in total_used_idx:
                        total_loss += o["loss_accum"]
                        total_n += o["n"]
                        total_y_true.extend(o["y_true"])
                        total_y_score.extend(o["y_score"])
                        total_used_idx.append(o["used_idx"])

                if self.rank == 0:
                    two_labels = len(np.unique(np.array(total_y_true)).tolist()) == 2
                    if two_labels:
                        _, auroc, acc, _ = calc(
                            np.array(total_y_true), np.array(total_y_score)
                        )

                        self.log(
                            f"loss: {total_loss / total_n:.4f} "
                            f"acc: {acc:.4f} auc: {auroc:.4f} "
                            f"iter: {i}/{len(self.val_ldr)} "
                            f"idx: {len(total_used_idx)}"
                        )

        return (total_loss / total_n, np.array(total_y_true), np.array(total_y_score))

    def pretrain_setup(self) -> None:
        """
        for pretraining of the head on the extracted features. This will not
        be a DDP instance.
        """
        torch.cuda.set_device(self.args.gpu)
        self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(0.5, 0.9),
            weight_decay=self.args.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.args.epochs, 5e-6
        )

        def get_train_loaders() -> None:
            self.trainset = CAMELYON16Patches256Extracted(
                root=self.args.data_root, split="train"
            )
            self.valset = CAMELYON16Patches256Extracted(
                root=self.args.data_root, split="test"
            )
            self.train_ldr = DataLoader(
                dataset=self.trainset,
                batch_size=self.args.batch_size,
                collate_fn=pad_collate,
                shuffle=True,
                num_workers=8,
            )
            self.val_ldr = DataLoader(
                dataset=self.valset,
                batch_size=self.args.batch_size,
                collate_fn=pad_collate,
                num_workers=6,
                shuffle=False,
            )
            self.log(f"{len(self.trainset)=} {len(self.valset)=}")

        def get_test_loader() -> None:
            self.valset = CAMELYON16Patches256Extracted(
                root=self.args.data_root, split="test"
            )
            self.log(f"{len(self.valset)=}")

            self.val_ldr = DataLoader(
                dataset=self.valset,
                batch_size=self.args.batch_size,
                collate_fn=pad_collate,
                num_workers=6,
                shuffle=False,
            )

        if self.args.mode == "pretrain":
            get_train_loaders()
        elif self.args.mode == "pretrain-test":
            get_test_loader()

    def pretrain_fit(self) -> None:
        while self.epoch < self.args.epochs:
            avg_loss, y_true, y_score = self.pretrain()
            aupr, auroc, acc, thresh = calc(y_true, y_score)
            stats_dict = {"acc": acc, "aupr": aupr, "auroc": auroc, "thresh": thresh}

            self.log_stats(self.results_path, "pretrain", stats_dict)

            avg_loss, y_true, y_score = self.pretrain_test()
            aupr, auroc, acc, thresh = calc(y_true, y_score)
            stats_dict = {"acc": acc, "aupr": aupr, "auroc": auroc, "thresh": thresh}

            self.log_stats(self.results_path, "pretrain-val", stats_dict)

            score = (acc + auroc) / 2
            if score > self.best_score:
                self.best_score = score

                self.log("saving model")
                self.save_model(self.models_path, "pretrain")

            self.epoch += 1
            self.scheduler.step()

        self.log("finished pretraining, saving...")
        self.load_model(self.models_path, "pretrain")
        self.save_model(self.models_path, "pretrain", finished=True)

    def pretrain_forward(self, x: T, mask: T, train: bool = True) -> Tuple[T, T]:
        model = self.get_model()
        x, mask = x.cuda(), mask.cuda()

        if not model.is_mbc:
            ins_pred, bag_pred = model.decoder(x.squeeze(0))
            ins_pred = ins_pred.view(-1).amax().view(1)
            bag_pred = bag_pred.view(-1)
            return ins_pred, bag_pred

        # model is mbc, do the mbc method
        model.pre_forward_mbc()

        chnk_ft, x_t, c_t, chunks = [], None, None, 1
        xchnk, mchnk = x.chunk(chunks, dim=1), mask.chunk(chunks, dim=1)
        for i, (_x, _mask) in enumerate(zip(xchnk, mchnk)):
            grad = train and (i == 0)
            x_ft, x_t, c_t = model.forward_mbc(
                _x, mask=_mask, X_prev=x_t, c_prev=c_t, grad=grad
            )
            chnk_ft.append(x_ft)

        assert isinstance(x_t, T)
        x = model.post_forward_mbc(x_t, c=c_t)

        ins_pred, bag_pred = model.decoder(torch.cat(chnk_ft, dim=1), x)
        ins_pred = ins_pred.view(-1).amax().view(1)
        bag_pred = bag_pred.view(-1)
        return ins_pred, bag_pred

    def pretrain(self) -> Tuple[float, A, A]:
        self.model.train()

        losses, y_true, y_score, n = 0.0, [], [], 0
        for i, (x, mask, y) in enumerate(self.train_ldr):
            ins_pred, bag_pred = self.pretrain_forward(x, mask, train=True)

            loss_ins = self.bce_logits(ins_pred, y.float().cuda())
            loss_bag = self.bce_logits(bag_pred, y.float().cuda())

            loss = (loss_ins + loss_bag) / 2
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                pred = (torch.sigmoid(ins_pred) + torch.sigmoid(bag_pred)) / 2

                losses += loss.item() * y.size(0)
                n += y.size(0)
                y_score.extend(pred.tolist())
                y_true.extend(y.tolist())

        return losses / n, np.array(y_true), np.array(y_score)

    def pretrain_test(self) -> Tuple[float, A, A]:
        self.model.eval()

        losses, y_true, y_score, n = 0.0, [], [], 0
        for i, (x, mask, y) in enumerate(self.val_ldr):
            with torch.no_grad():
                ins_pred, bag_pred = self.pretrain_forward(x, mask, train=True)

                loss_ins = self.bce_logits(
                    ins_pred, y.repeat(ins_pred.size(0)).float().cuda()
                )
                loss_bag = self.bce_logits(bag_pred, y.float().cuda())

                loss = (loss_ins + loss_bag) / 2

                pred = (torch.sigmoid(ins_pred) + torch.sigmoid(bag_pred)) / 2

                losses += loss.item() * y.size(0)
                n += y.size(0)
                y_score.extend(pred.tolist())
                y_true.extend(y.tolist())

        return losses / n, np.array(y_true), np.array(y_score)

    def load_model(self, path: str, addendum: str) -> None:  # type: ignore
        model_path = os.path.join(path, f"{self.args.model_hash}-{addendum}.pt")
        if self.is_ddp():
            raise ValueError("cannot load model when DDP is already instantiated")

        self.log(f"MODEL PATH: {model_path=}")
        if os.path.exists(model_path):
            saved = torch.load(model_path, map_location="cpu")
            self.epoch = saved["epoch"]
            self.best_score = saved["best_score"]

            self.model.load_state_dict(saved["state_dict"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.scheduler.load_state_dict(saved["scheduler"])

            self.finished = saved["finished"]
            self.tuned = saved.get("tuned", False)
            self.loaded = True
            self.log(
                f"loaded saved model: {self.epoch=} {self.finished=}\nfrom path: {model_path}"
            )  # noqa

    def save_model(  # type: ignore
        self, path: str, addendum: str, finished: bool = False
    ) -> None:
        sd_path = os.path.join(path, f"{self.args.model_hash}-{addendum}.pt")
        save = dict(
            epoch=self.epoch,
            state_dict=self.get_model().state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            finished=finished,
            best_score=self.best_score,
            tuned=self.tuned,
        )
        if self.model_ema is not None:
            save["model_ema"] = self.model_ema.ema.state_dict()
        torch.save(save, sd_path)

    def get_model(self) -> Model:
        if isinstance(self.model, DDP):
            return self.model.module  # type: ignore
        return self.model

    def get_results_keys(self, additional_keys: Dict[str, Any] = {}) -> Dict[str, Any]:
        keys = {"model": self.get_model().name}
        if isinstance(self.get_model(), HashableModule):
            keys = self.get_model().get_results_keys()

        return {
            **keys,
            "epoch": self.epoch,
            "model_hash": self.args.model_hash,
            "run": self.args.run,
            "grad_set_size": self.args.grad_set_size,
            "grad_correction": self.args.grad_correction,
            "comment": self.args.comment,
        }

    def print_log_stats(
        self,
        prefix: str,
        stats_dict: Dict[str, Any],
    ) -> None:
        msg = f"({prefix}-{self.get_model().name} "
        msg += f"epoch: {self.epoch}/{self.args.epochs} "
        msg += f"k: {self.args.k} "
        for _, (n, v) in enumerate(stats_dict.items()):
            msg += f"{n}: {v:.4f} "

        self.log(msg)

    def log_stats_df(
        self, path: str, info_dict: Dict[str, Any], stats_dict: Dict[str, Any]
    ) -> None:

        data: Dict[str, Any] = {n: [v] for (n, v) in stats_dict.items()}
        data["timestamp"] = str(datetime.now())
        for k in info_dict:
            data[k] = [info_dict[k]]

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(data)
        with open(path, "a+") as f:
            # only write header if the current line is 0
            df.to_csv(f, mode="a", header=f.tell() == 0)

    def log_stats(self, path: str, prefix: str, stats_dict: Dict[str, Any]) -> None:
        result_keys = {**self.get_results_keys()}
        path = os.path.join(path, f"{prefix}-results.csv")
        self.log_stats_df(path, result_keys, stats_dict)

        self.print_log_stats(prefix, stats_dict)

    def log(self, msg: str) -> None:
        self.args.logger.info(msg)

    def fit(self) -> None:
        raise NotImplementedError("camelyon trainer uses special fitting routine")

    def train(self) -> None:
        raise NotImplementedError("camelyon trainer uses special training routine")

    def test(self) -> None:
        raise NotImplementedError("camelyon trainer uses special training routine")

    def log_train_stats(self, path: str) -> Dict[str, float]:
        """logs train/val stats periodically during training"""
        raise NotImplementedError()

    def log_test_stats(self, path: str, test_name: str = "test") -> Dict[str, float]:
        """logs test stats including a test name (test, corrupt, etc...)"""
        raise NotImplementedError()


def setup(rank: int, world_size: int, run: int) -> None:
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '1235' + str(run)
    dist_url = "tcp://127.0.0.1:1234" + str(run)
    # initialize the process group
    dist.init_process_group(
        backend="nccl", init_method=dist_url, rank=rank, world_size=world_size
    )


def cleanup() -> None:
    dist.destroy_process_group()


def signal_handler(sig: Any, frame: Any) -> None:
    print("\nquitting gracefully\n")
    cleanup()
    sys.exit(0)


if __name__ == "__main__":
    parser = ArgumentParser("argument parser for MBC Camelyon")

    model_choices = [
        "deepsets",
        "deepsets-umbc",
        "sse",
        "sse-umbc",
        "ab-mil",
        "ds-mil",
        "sse-fspool",
    ]
    mode_choices = ["finetune", "pretrain", "pretrain-test", "finetune-test"]
    attn_acts = ["sigmoid", "softmax", "slot-sigmoid", "slot-softmax", "slot-exp"]

    parser.add_argument(
        "--dataset",
        type=str,
        default="camelyon-patches",
        choices=["camelyon-patches"],
        help="the dataset to use",
    )
    # parser.add_argument("--data-root", type=str,
    #                     default="/d1/dataset", help="dataset root"),
    # parser.add_argument("--data-root", type=str,
    #                     default="/w17/camelyon", help="dataset root"),
    parser.add_argument("--data-root", type=str, default="/data", help="dataset root"),
    parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="comment to add to the hash string/results file",
    )
    parser.add_argument("--run", type=int, default=0, help="run number")
    parser.add_argument("--num-workers", type=int, default=1, help="run number")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="batch size for training"
    )
    parser.add_argument("--h-dim", type=int, default=128, help="hidden dim")
    parser.add_argument(
        "--pool",
        type=str,
        default="mean",
        choices=["mean", "min", "max", "sum"],
        help="pooling function if relevant",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="the set of GPUs to distribute the models to",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="the GPU to use for pretraining"
    )
    parser.add_argument(
        "--heads", type=int, default=1, help="number of attention heads"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="initial learning rate")
    parser.add_argument("--mode", type=str, choices=mode_choices)
    parser.add_argument(
        "--slot-type", type=str, default="random", choices=["random", "deterministic"]
    )
    parser.add_argument(
        "--ln-slots",
        type=str2bool,
        default=True,
        help="put layernorm pre-activation on the slots?",
    )
    parser.add_argument(
        "--grad-correction",
        type=str2bool,
        default=True,
        help="add grad subset gradient correction to model",
    )
    parser.add_argument(
        "--slot-residual",
        type=str2bool,
        default=True,
        help="put a residual connection on the SSE output",
    )
    parser.add_argument("--model", type=str, choices=model_choices)
    parser.add_argument(
        "--attn-act",
        type=str,
        choices=attn_acts,
        default="slot-sigmoid",
        help="the attention activation on MBC models",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="the number of epochs to run for"
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=10,
        help="the number of epochs tofinetune for",
    )
    parser.add_argument(
        "--ln-after", type=str2bool, default=True, help="put layernorm after SSE"
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--k",
        type=int,
        default=128,
        help="the number of slots to use in the sse encoder",
    )
    parser.add_argument(
        "--grad-set-size",
        type=int,
        default=32,
        help="the size of the partitioned gradient",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=5.0, help="grad clipping threshold"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="the size of the partitioned gradient",
    )
    parser.add_argument(
        "--patch-dropout",
        type=float,
        default=0.0,
        help="dropout rate for patches during finetuning",
    )
    parser.add_argument("--augmentation", type=str2bool, default=False)
    parser.add_argument("--linear", type=str2bool, default=False)
    parser.add_argument("--model_ema", type=str2bool, default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.98, help="")
    args = parser.parse_args()
    args.logger = set_logger("INFO")

    # seed before doing anything else with the dataloaders
    seed(args.run)

    out_dim = 1
    decoder_depth = 1
    func_deref = {
        "deepsets": partial(
            CamelyonDeepSets,
            out_dim=out_dim,
            n_decoder_layers=decoder_depth,
            h_dim=128,
            pool=args.pool,
            use_resnet="finetune" in args.mode,
        ),
        "sse": partial(
            CamelyonSSE,
            out_dim=out_dim,
            n_decoder_layers=decoder_depth,
            h=args.h_dim,
            d=256,
            d_hat=args.h_dim,
            heads=args.heads,
            slot_drop=0.0,
            attn_act=args.attn_act,
            slot_residual=args.slot_residual,
            ln_after=args.ln_after,
            use_resnet="finetune" in args.mode,
        ),
        "sse-umbc": partial(
            CamelyonSSEUMBC,
            K=args.k,
            h=args.h_dim,
            d=256,
            d_hat=args.h_dim,
            out_dim=out_dim,
            n_decoder_layers=decoder_depth,
            heads=args.heads,
            slot_drop=0.0,
            attn_act=args.attn_act,
            slot_residual=args.slot_residual,
            ln_after=args.ln_after,
            use_resnet="finetune" in args.mode,
        ),
        # "sse-fspool": partial(
        #     CamelyonSSEFSPool, K=args.k, h=args.h_dim, d=256,
        #     d_hat=args.h_dim, out_dim=out_dim, n_decoder_layers=decoder_depth,
        #     heads=args.heads, slot_drop=0.0,
        #     attn_act=args.attn_act, slot_residual=args.slot_residual,
        #     ln_after=args.ln_after,
        #     use_resnet="finetune" in args.mode,
        # ),
        "ab-mil": partial(
            CamelyonABMIL,
            out_dim=out_dim,
            h_dim=args.h_dim,
            use_resnet="finetune" in args.mode,
        ),
        "ds-mil": partial(
            CamelyonDSMIL,
            out_dim=out_dim,
            h_dim=args.h_dim,
            use_resnet="finetune" in args.mode,
        ),
    }
    model = func_deref[args.model]()  # type: ignore

    # at pre-training, we used a grad size parameter of 128 even though
    # the grad set size is not used during pretraining. Therefore, it
    # went into the model hash. In order to finetune with a larger
    # grad set size, it was necessary to hardcode this to be 128
    # for loading the model and then just use whatever grad set size
    # we want to during finetuning.
    train_grad_size = 128

    def get_string_and_hash(model: Any) -> Tuple[str, str]:
        model_string = f"GRAD SIZE: {train_grad_size}\n"
        model_string += f"COMMENT: {args.comment}\nRUN: {args.run}\n"
        model_string += f"GRAD CORRECTION: {args.grad_correction}"
        model_string += "\n\n" + str(model)
        model_hash = md5(model_string)
        return model_string, model_hash

    s, h = get_string_and_hash(model)
    args.model_hash = h
    args.model_string = s

    trainer = CamelyonTrainer(args, model)

    def pretrain() -> None:
        trainer.pretrain_setup()
        trainer.load_model(trainer.models_path, "pretrain")
        if trainer.finished:
            trainer.log("called fit() on a model which has finished training")
            exit()

        trainer.pretrain_fit()

    def pretrain_test() -> None:
        trainer.pretrain_setup()
        trainer.load_model(trainer.models_path, "pretrain")
        if not trainer.finished:
            raise ValueError("cannot test a model which has not finished training.")

        avg_loss, y_true, y_score = trainer.pretrain_test()
        aupr, auroc, acc, thresh = calc(y_true, y_score)
        stats_dict = {"acc": acc, "aupr": aupr, "auroc": auroc, "thresh": thresh}

        trainer.log_stats(trainer.results_path, "pretrain-test", stats_dict=stats_dict)

    if args.mode == "pretrain":
        pretrain()
        trainer.args.mode = "pretrain-test"
        pretrain_test()
    elif args.mode == "pretrain-test":
        pretrain_test()
    elif args.mode == "finetune":
        # set the extractor so we can load the correct pretrained model
        # given the model hash at pretraining (hash without extractor)
        model.extractor = IdentityExtractor()
        _, model_hash = get_string_and_hash(model)

        model_path = os.path.join(trainer.models_path, f"{model_hash}-pretrain.pt")

        if not os.path.exists(model_path):
            raise ValueError(f"no pretrained model found: {model_path}")

        trainer.log(f"MODEL PATH: {model_path=}")
        saved = torch.load(model_path, map_location="cpu")
        trainer.best_score = saved["best_score"]
        trainer.model.load_state_dict(saved["state_dict"])
        trainer.loaded = True
        trainer.log(f"loaded saved model from path: {model_path}")

        trainer.model.extractor = get_extractor(resnet=True)

        signal.signal(signal.SIGINT, signal_handler)
        mp.spawn(trainer.finetune_setup, nprocs=len(args.gpus), join=True)

        # after finetuneing, we should have a saved model with the hash which
        # contains the resnet in the model string. load that model and continue
        # with the testing
        trainer.args.mode = "finetune-test"
        trainer.model = model

        # there is no optimizer in the trainer here because the setup
        # was called in the multiprocessing, so we must load everything
        # manually
        _, model_hash = get_string_and_hash(model)
        model_path = os.path.join(trainer.models_path, f"{model_hash}-finetune.pt")
        if not os.path.exists(model_path):
            raise ValueError(f"no finetuned model found: {model_path}")

        trainer.log(f"MODEL PATH: {model_path=}")
        saved = torch.load(model_path, map_location="cpu")
        trainer.best_score = saved["best_score"]
        trainer.model.load_state_dict(saved["state_dict"])
        trainer.loaded = True
        if "model_ema" in saved and trainer.model_ema is not None:
            trainer.model_ema.ema.load_state_dict(saved["model_ema"])

        trainer.log(f"loaded saved model from path: {model_path}")

        mp.spawn(trainer.finetune_setup, nprocs=len(args.gpus), join=True)
    elif args.mode == "finetune-test":
        signal.signal(signal.SIGINT, signal_handler)

        # there is no optimizer in the trainer here, so we must load
        # everything manually
        _, model_hash = get_string_and_hash(model)
        model_path = os.path.join(trainer.models_path, f"{model_hash}-finetune.pt")
        if not os.path.exists(model_path):
            raise ValueError(f"no finetuned model found: {model_path}")

        trainer.log(f"MODEL PATH: {model_path=}")
        saved = torch.load(model_path, map_location="cpu")
        trainer.best_score = saved["best_score"]
        trainer.model.load_state_dict(saved["state_dict"])
        trainer.loaded = True
        trainer.log(f"loaded saved model from path: {model_path}")

        mp.spawn(trainer.finetune_setup, nprocs=len(args.gpus), join=True)
    else:
        raise NotImplementedError(f"mode: {args.mode} is not valid")
