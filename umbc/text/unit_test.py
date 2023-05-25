import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from umbc.models.text_classifier import MBC
from utils import seed, str2bool

from data.get import get_dataset

NUM_CLASSES = {"cmu_book_summaries": 227,
               "eurlex": 4271}


def set_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


class UnitTest(object):
    def __init__(self,
                 args: Namespace,
                 model: nn.Module,
                 ):

        self.args = args
        self.model = model.to(self.args.device)

    def test(self, dataloader: DataLoader):
        self.model.train()
        device = self.args.device
        num_errors = 0
        for batch in tqdm(dataloader, ncols=75, leave=False):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, labels = batch
            inputs = {"input_ids": input_ids,
                      "input_mask": input_mask,
                      "labels": labels}
            seed = random.randint(0, 9999)
            set_seed(seed)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs[1]

            set_seed(seed)
            inputs["split_size"] = 200
            with torch.no_grad():
                outputs = self.model.forward_mbc(**inputs)
                logits_mbc = outputs[1]
            err = torch.mean((logits-logits_mbc).abs()).item()
            if err > 1e-3:
                num_errors += 1
            print(err)
        print("num failures:", num_errors)


if __name__ == "__main__":
    parser = ArgumentParser("argument parser for MBC Text Classification")
    parser.add_argument("--gpu", type=str, default="0", help="the gpu index")
    parser.add_argument("--mode", type=str, choices=["train", "test"])
    parser.add_argument("--dataset", type=str, default="eurlex",
                        choices=["cmu_book_summaries", "eurlex"], help="the dataset to use")
    parser.add_argument("--data_root", type=str,
                        default="code/dataset")
    parser.add_argument("--num_workers", type=int, default=8)

    # encoder
    parser.add_argument("--num_layers", type=int, default=2)

    parser.add_argument("--comment", type=str, default="",
                        help="comment to add to the hash string and the results file")
    parser.add_argument("--run", type=int, default=0, help="run number")
    parser.add_argument("--epochs", type=int, default=20,
                        help="the number of epochs to run for")
    parser.add_argument("--batch-size", type=int,
                        default=16, help="batch size for training")
    parser.add_argument("--scheduler", default=False, type=str2bool)
    parser.add_argument("--train_set_size", type=int,
                        default=-1, help="set size for training")
    parser.add_argument("--test_set_size", type=int,
                        default=-1, help="set size for test")
    #
    parser.add_argument("--agg", type=str2bool, default=False)
    parser.add_argument("--umbc_grad_size", type=int, default=100,
                        help="the size of the partitioned umbc gradient")

    # optimizer
    parser.add_argument("--weight_decay", type=float, default=5e-3)
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

    args = parser.parse_args()
    args.device = torch.cuda.current_device()
    args.test_set_size = args.train_set_size

    if args.agg:
        args.comment = "agg"
    seed(args.run)
    # if args.mode == "train":
    train_ldr, val_ldr, test_ldr = get_dataset(args)
    # else:
    #     train_ldr, val_ldr, test_ldr = None, None, None
    model = MBC(
        num_layers=args.num_layers,
        d_dim=args.d, h_dim=args.h,
        d_hat=args.d_hat, K=args.k,
        heads=args.heads,
        num_classes=NUM_CLASSES[args.dataset],
        ln_slots=args.ln_slots, ln_after=args.ln_after,
        slot_type=args.slot_type, slot_drop=args.slot_drop,
        attn_act=args.attn_act, slot_residual=args.slot_residual)

    trainer = UnitTest(args, model)
    trainer.test(train_ldr)
