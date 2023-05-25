import os
from argparse import ArgumentParser, Namespace
from typing import Any

import numpy as np
import torch
from data.celeba import CELEBAResized
from data.get import get_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from utils import seed, str2bool

from umbc.models.cnp import MBC, UMBC, DeepSet, SetTransformer

T = torch.Tensor
SetEncoder = nn.Module


class InfIterator(object):
    def __init__(self, iterable) -> Any:
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)


class CelebATrainer(object):
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.train_steps)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        # optimizer, [int(r*args.train_steps) for r in [0.8]], gamma=0.1)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.train_iter = InfIterator(self.train_loader)

        exp_name = f"{args.dataset}_{model.name}_{args.train_set_size}_{args.comment}"

        self.ckpt_path = os.path.join(
            "checkpoints/celeba", f"{exp_name}_{args.run}")
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        if self.args.mode == "test":
            exp_name += "test"

    def fit(self) -> None:
        t = tqdm(range(1, self.args.train_steps + 1), ncols=75, leave=False)
        train_iter = InfIterator(self.train_loader)
        for global_step in t:
            self.model.train()
            batch = next(train_iter)
            batch = tuple(t.to(self.args.device) for t in batch)
            context, target_x, target_y = batch

            log_prob = self.forward(context, target_x, target_y)
            loss = -log_prob.mean()

            self.model.zero_grad()
            loss.backward()
            if self.args.agg:
                c = self.args.train_set_size / self.args.umbc_grad_size
                self.model.sse_grad_correct(c)

            t.set_description(
                "Step: {} / {}, loss: {:.4f}".format(global_step, self.args.train_steps, loss.item()))

            self.optimizer.step()
            self.scheduler.step()

            if global_step % 5000 == 0:
                valid_loss = self.test(self.valid_loader)
        test_loss = self.test(self.test_loader)

        self.save_model(self.ckpt_path)

    def forward(self, context: T, target_x: T, target_y: T) -> T:
        if self.args.agg:
            idx = torch.randperm(context.size(1))
            c = context[:, idx[:self.args.umbc_grad_size]]
            c_nograd = context[:, idx[self.args.umbc_grad_size:]]
            # type: ignore
            return self.model.partitioned_forward(c, c_nograd, target_x, target_y)
        else:
            return self.model(context, target_x, target_y)

    def test(self, dataloader: DataLoader):
        self.model.eval()
        device = self.args.device
        all_loss = []
        for batch in tqdm(dataloader, ncols=75, leave=False):
            batch = tuple(t.to(device) for t in batch)
            context, target_x, target_y = batch
            with torch.no_grad():
                ll = self.model(context, target_x, target_y)
            loss = -ll.mean()
            all_loss.append(loss.item())
        return np.mean(all_loss)

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
    parser = ArgumentParser("argument parser for MBC CelebA")
    parser.add_argument("--gpu", type=str, default="0", help="the gpu index")
    parser.add_argument("--mode", type=str, choices=["train", "test"])
    parser.add_argument("--dataset", type=str, default="celeba",
                        choices=["celeba"], help="the dataset to use")
    parser.add_argument("--data_root", type=str, default="/d1/dataset/")
    parser.add_argument("--num_workers", type=int, default=8)
    # CNP
    parser.add_argument("--x_dim", type=int, default=2)
    parser.add_argument("--y_dim", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=4)

    parser.add_argument("--comment", type=str, default="",
                        help="comment to add to the hash string and the results file")
    parser.add_argument("--run", type=int, default=0, help="run number")

    parser.add_argument("--train_steps", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=100,
                        help="the number of epochs to run for")
    parser.add_argument("--batch-size", type=int,
                        default=256, help="batch size for training")

    parser.add_argument("--train_set_size", type=int,
                        default=100, help="set size for training")
    parser.add_argument("--test_set_size", type=int,
                        default=100, help="set size for test")
    #
    parser.add_argument("--agg", type=str2bool, default=False)
    parser.add_argument("--umbc_grad_size", type=int, default=100,
                        help="the size of the partitioned umbc gradient")

    # optimizer
    parser.add_argument("--lr", type=float, default=5e-4)

    # set encoder
    parser.add_argument("--model", type=str, default="mbc")
    parser.add_argument("--d", type=int, default=128,
                        help="sdim for encoder/decoder")
    parser.add_argument("--k", type=int, default=1, help="the number of slots")
    parser.add_argument("--h", type=int, default=128,
                        help="hidden_dim for slot")
    parser.add_argument("--d_hat", type=int, default=128,
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
                        "slot-softmax", "slot-exp"], default="softmax")

    args = parser.parse_args()
    args.device = torch.cuda.current_device()
    args.test_set_size = args.train_set_size

    if args.agg:
        args.comment = "agg"
    seed(args.run)
    if args.mode == "train":
        train_ldr, val_ldr, test_ldr = get_dataset(args)
    else:
        train_ldr, val_ldr, test_ldr = None, None, None
    model_deref = {
        "mbc": MBC,
        "umbc": UMBC,
        "deepset": DeepSet,
        "transformer": SetTransformer
    }
    model = model_deref[args.model](x_dim=args.x_dim, y_dim=args.y_dim, d_dim=args.d,
                                    h_dim=args.h, d_hat=args.d_hat,
                                    K=args.k, e_depth=args.num_layers,
                                    d_depth=args.num_layers, heads=args.heads,
                                    ln_slots=args.ln_slots, ln_after=args.ln_after,
                                    slot_type=args.slot_type, slot_drop=args.slot_drop,
                                    attn_act=args.attn_act, slot_residual=args.slot_residual)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = CelebATrainer(args, model, opt, train_ldr, val_ldr, test_ldr)

    if args.mode == "train":
        trainer.fit()
    else:
        trainer.load_model()
        for num_points_eval in [100, 200, 300, 400, 500]:
            dataset = CELEBAResized(args.data_root,
                                    split="test",
                                    num_points=num_points_eval,
                                    num_points_eval=num_points_eval)
            dataset = [dataset[i] for i in tqdm(range(len(dataset)))]
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
            loss = trainer.test(dataloader)
            print("{}: NLL: {:.4f}".format(num_points_eval, loss))
