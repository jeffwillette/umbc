import math
import os
import random
from argparse import ArgumentParser, Namespace
from typing import Any

import numpy as np
import torch
from data.get import get_dataset
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from transformers.optimization import get_linear_schedule_with_warmup
from utils import seed, str2bool

from umbc.models.text_classifier import MBC, UMBC, BoW

NUM_CLASSES = {"cmu_book_summaries": 227,
               "eurlex": 4271}


def set_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


class TextTrainer(object):
    def __init__(self,
                 args: Namespace,
                 model: nn.Module,
                 optimizer: Any,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 test_loader: DataLoader):

        self.args = args
        self.model = model.to(self.args.device)

        self.optimizer = optimizer
        if args.scheduler:
            num_steps = self.epochs * len(self.trainloader)
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_steps * 0.1, num_training_steps=num_steps)
        else:
            self.scheduler = None
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        exp_name = f"{args.dataset}_{model.name}_{args.train_set_size}_{args.epochs}_{args.comment}_pretrain_{args.pretrain}"
        if args.agg:
            exp_name += f"_{args.umbc_grad_size}"
        exp_name += f"_BS_{args.batch_size}"

        if self.args.mode == "test":
            exp_name += "test"


    def fit(self) -> None:
        global_step = 1
        valid_f1 = -1
        for epoch in range(1, self.args.epochs+1):
            with tqdm(total=len(self.train_loader), ncols=75, leave=False) as t:
                self.model.train()
                for batch in self.train_loader:
                    batch = tuple(t.to(self.args.device) for t in batch)
                    input_ids, input_mask, labels = batch
                    loss, f1 = self.forward(input_ids, input_mask, labels)
                    
                    t.set_description(
                        "EPOCH: {}, step: {}, loss: {:.4f},train f1: {:.4f} valid f1: {:.4f}".format(epoch, global_step, loss, f1, valid_f1))
                    t.update(1)

                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    global_step += 1
            valid_loss, valid_f1 = self.test(self.valid_loader)
            
        self.reset_opt()
        # for n,p in self.model.decoder.named_parameters():
        #     print(n, p.requires_grad)
        for epoch in range(1, self.args.epochs+1):
            with tqdm(total=len(self.train_loader), ncols=75, leave=False) as t:
                self.model.train()
                for batch in self.train_loader:
                    batch = tuple(t.to(self.args.device) for t in batch)
                    input_ids, input_mask, labels = batch
                    loss, f1 = self.forward(input_ids, input_mask, labels)

                    
                    t.set_description(
                        "EPOCH: {}, step: {}, loss: {:.4f}, train f1: {:.4f} valid f1: {:.4f}".format(epoch, global_step, loss, f1, valid_f1))
                    t.update(1)

                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    global_step += 1
            valid_loss, valid_f1 = self.test(self.valid_loader)
            
        test_loss, test_f1 = self.test(self.test_loader)
        
    def reset_opt(self):
        # finetune
        if self.args.finetune:
            for p in self.model.parameters():
                p.requires_grad = True
        params = list(filter(lambda p: p.requires_grad,
                      self.model.parameters()))
        self.optimizer = torch.optim.AdamW(
            params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.scheduler is not None:
            num_steps = self.epochs * len(self.trainloader)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_steps * 0.1, num_training_steps=num_steps)

    def forward(self, input_ids, input_mask, labels):
        inputs = {"input_ids": input_ids,
                  "input_mask": input_mask,
                  "labels": labels}
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        if self.args.agg:
            inputs["split_size"] = self.args.umbc_grad_size
            # set seed
            seed = random.randint(0, 9999)
            set_seed(seed)

            outputs = self.model.forward_mbc(**inputs)
            loss = outputs[0]
            model.zero_grad()
            loss.backward()

            train_loss = loss.item()

            params = list(self.model.decoder.parameters()) \
                + list(self.model.pooler.norm_after.parameters())
            buffers = [torch.zeros_like(p) for p in params]
            for b, p in zip(buffers, params):
                if p.grad is not None:
                    b.copy_(p.grad.data)

            inputs["labels"] = None
            set_seed(seed)

            self.model.zero_grad()
            output = self.model.forward_mbc(**inputs)
            logits = output[0]
            ind_loss = criterion(logits, labels)

            lengths = torch.sum(input_mask, 1).float()
            num_chunks = torch.tensor([math.ceil(l / self.args.umbc_grad_size)
                                       for l in lengths])
            num_chunks = num_chunks.unsqueeze(1).to(self.args.device)
            loss = (num_chunks * ind_loss).mean()
            loss.backward()

            params = list(self.model.decoder.parameters()) \
                + list(self.model.pooler.norm_after.parameters())
            for buffer, p in zip(buffers, params):
                if p.grad is not None:
                    p.grad.copy_(buffer)

        else:
            outputs = self.model(**inputs)
            loss, logits = outputs[0], outputs[1]
            self.model.zero_grad()
            loss.backward()

            train_loss = loss.item()

        probs = torch.sigmoid(logits).detach()
        preds = torch.where(probs > 0.5, 1, 0).cpu().numpy()
        labels = labels.detach().cpu().numpy()

        f1 = f1_score(labels, preds, average="micro")

        return train_loss, f1

    def test(self, dataloader: DataLoader):
        self.model.eval()
        device = self.args.device
        all_loss = []
        all_labels = []
        all_preds = []
        for batch in tqdm(dataloader, ncols=75, leave=False):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            inputs = {"input_ids": input_ids,
                      "input_mask": input_mask,
                      "labels": labels}
            with torch.no_grad():
                outputs = self.model(**inputs)
            loss = outputs[0]
            logits = outputs[1]

            probs = torch.sigmoid(logits)
            preds = torch.where(probs > 0.5, 1, 0)
            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())

            all_loss.append(loss.item())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        f1 = f1_score(all_labels, all_preds, average="micro")

        return np.mean(all_loss), f1

    def save_model(self, path: str):
        ckpt = {"args": self.args,
                "state_dict": self.model.state_dict()}

        ckpt_file = os.path.join(path, "model.pt")
        torch.save(ckpt, ckpt_file)
        print("model saved")

    def load_model(self):

        ckpt = torch.load(os.path.join(self.ckpt_path, "model.pt"))
        state_dict = ckpt["state_dict"]
        print("load model")
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    parser = ArgumentParser("argument parser for MBC Text Classification")
    parser.add_argument("--gpu", type=str, default="0", help="the gpu index")
    parser.add_argument("--mode", type=str, choices=["train", "test"])
    parser.add_argument("--dataset", type=str, default="cmu_book_summaries",
                        choices=["cmu_book_summaries", "eurlex"], help="the dataset to use")
    parser.add_argument("--data_root", type=str,
                        default="code/dataset/")
    parser.add_argument("--num_workers", type=int, default=8)

    # model selection
    parser.add_argument("--bow", type=str2bool, default=False)
    parser.add_argument("--sse", type=str2bool, default=False)
    parser.add_argument("--add_st", type=str2bool, default=False)

    # encoder
    parser.add_argument("--num_layers", type=int, default=2)

    parser.add_argument("--comment", type=str, default="",
                        help="comment to add to the hash string and the results file")
    parser.add_argument("--run", type=int, default=0, help="run number")

    parser.add_argument("--epochs", type=int, default=20,
                        help="the number of epochs to run for")
    parser.add_argument("--batch-size", type=int,
                        default=8, help="batch size for training")
    parser.add_argument("--scheduler", default=False, type=str2bool)
    parser.add_argument("--train_set_size", type=int,
                        default=100, help="set size for training")
    parser.add_argument("--test_set_size", type=int,
                        default=100, help="set size for test")
    #
    parser.add_argument("--agg", type=str2bool, default=False)
    parser.add_argument("--umbc_grad_size", type=int, default=100,
                        help="the size of the partitioned umbc gradient")
    parser.add_argument("--detach_c", type=str2bool, default=False)
    # optimizer
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--opt", type=str, default="adamw")
    parser.add_argument("--finetune", type=str2bool, default=False)

    # set encoder
    parser.add_argument("--model", type=str, default="mbc")
    parser.add_argument("--d", type=int, default=768,
                        help="sdim for encoder/decoder")
    parser.add_argument("--k", type=int, default=256,
                        help="the number of slots")
    parser.add_argument("--h", type=int, default=128,
                        help="hidden_dim for slot")
    parser.add_argument("--d_hat", type=int, default=768,
                        help="dim after k,q,v projection")
    parser.add_argument("--heads", type=int, default=1,
                        help="number of attention heads")
    parser.add_argument("--slot_type", type=str,
                        default="random", choices=["random", "deterministic"])
    parser.add_argument("--ln_slots", type=str2bool, default=True,
                        help="put layernorm pre-activation on the slots?")
    parser.add_argument("--grad_correction", type=str2bool, default=True,
                        help="whether or not to add a gradient correction to umbc grad subset")
    parser.add_argument("--ln_after", type=str2bool,
                        default=True, help="put layernorm after SSE")
    parser.add_argument("--fixed", type=str2bool, default=False,
                        help="whether or not to fix the universal SSE weights")
    parser.add_argument("--slot_residual", type=str2bool, default=False,
                        help="whether or not to put a residual connection on the SSE output")
    parser.add_argument("--slot_drop", type=float, default=0.0,
                        help="slot dropout rate for the universal MBC models")
    parser.add_argument("--attn_act", type=str, choices=["sigmoid", "softmax", "slot-sigmoid",
                        "slot-softmax", "slot-exp"], default="slot-sigmoid")
    parser.add_argument("--pretrain", type=str2bool, default=True)
    args = parser.parse_args()
    args.device = torch.cuda.current_device()
    args.test_set_size = args.train_set_size

    if args.agg:
        args.comment = "agg"
    if args.detach_c:
        args.comment += "_detach"
        print("no detach")
        exit()
    seed(args.run)
    train_ldr, val_ldr, test_ldr = get_dataset(args)

    if args.bow:
        model = BoW(num_layers=args.num_layers)

    elif args.sse:
        print("SSE model")
        model = MBC(
            num_layers=args.num_layers,
            d_dim=args.d, h_dim=args.h,
            d_hat=args.d_hat, K=args.k,
            heads=args.heads,
            num_classes=NUM_CLASSES[args.dataset],
            ln_slots=args.ln_slots, ln_after=args.ln_after,
            slot_type=args.slot_type, slot_drop=args.slot_drop,
            attn_act=args.attn_act, add_st=args.add_st)

    else:
        print("UMBC model")
        model = UMBC(
            num_layers=args.num_layers,
            d_dim=args.d, h_dim=args.h,
            d_hat=args.d_hat, K=args.k,
            heads=args.heads,
            num_classes=NUM_CLASSES[args.dataset],
            ln_slots=args.ln_slots, ln_after=args.ln_after,
            slot_type=args.slot_type, slot_drop=args.slot_drop,
            attn_act=args.attn_act, slot_residual=args.slot_residual)

    for n, p in model.decoder.named_parameters():
        if "bert" in n:
            p.requires_grad = False
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    opt = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay)

    trainer = TextTrainer(args, model, opt, train_ldr, val_ldr, test_ldr)
    trainer.fit()
