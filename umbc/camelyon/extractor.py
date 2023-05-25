import os
import signal
import sys
from argparse import ArgumentParser, Namespace
from typing import Any, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.camelyon import CAMELYON16Patches256Iterable
from data.get import list_collate
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils import set_logger

from umbc.models.layers.resnet import ResNet50Pretrained, ResNetSimCLR
from umbc.models.layers.resnet_trunc import resnet50_trunc_baseline

T = torch.Tensor


def setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


def signal_handler(sig: Any, frame: Any) -> None:
    print('\nquitting gracefully\n')
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class CamelyonExtractor:
    def __init__(
        self,
        args: Namespace,
        model: nn.Module,
        trainset: Dataset,
        testset: Dataset
    ):
        super().__init__()

        self.args = args
        self.model: Union[nn.Module, DDP] = model

        self.trainset = trainset
        self.testset = testset

        self.train_ldr: DataLoader
        self.test_ldr: DataLoader

        self.train_sampler: DistributedSampler
        self.test_sampler: DistributedSampler

        self.rank: int

        self.features_path = os.path.join("/data", "extracted")
        os.makedirs(self.features_path, exist_ok=True)

    def is_ddp(self) -> bool:
        if isinstance(self.model, DDP):
            if self.rank != 0:
                raise ValueError(
                    f"is ddp can only be called on rank 0 model: {self.rank=}")
            return True
        return False

    def distributed_fit(self, rank: int) -> None:
        self.model = self.model.to(rank)
        self.rank = rank
        self.args.logger = set_logger("INFO")
        torch.cuda.set_device(self.args.gpus[self.rank])

        # load model must be handled in the main function
        setup(self.rank, len(self.args.gpus))

        self.model = DDP(self.model.cuda(), find_unused_parameters=True)

        self.train_sampler = DistributedSampler(
            self.trainset, num_replicas=len(self.args.gpus),
            rank=rank, shuffle=False)

        self.test_sampler = DistributedSampler(
            self.testset, num_replicas=len(self.args.gpus),
            rank=rank, shuffle=False)

        self.train_ldr = DataLoader(
            dataset=self.trainset,
            batch_size=1,
            collate_fn=list_collate,
            num_workers=0,
            shuffle=False,
            sampler=self.train_sampler
        )
        self.test_ldr = DataLoader(
            dataset=self.testset,
            batch_size=1,
            collate_fn=list_collate,
            num_workers=0,
            sampler=self.test_sampler,
            shuffle=False,
        )

        self.extract(self.train_ldr)
        self.extract(self.test_ldr)

    def extract(self, ldr: DataLoader) -> None:
        self.model.eval()

        for i, (batch_x, batch_y) in enumerate(ldr):
            for j, (x_loader, y) in enumerate(zip(batch_x, batch_y)):
                with torch.no_grad():
                    o = [self.model(v.cuda()) for v in x_loader]
                    out = torch.cat([v.cpu() for v in o])

                    mag, name = x_loader.dataset.files[0].split("/")[4:6]
                    self.log(f"{mag=} {name=}")
                    path = os.path.join(self.features_path, mag)
                    os.makedirs(path, exist_ok=True)
                    np_path = os.path.join(path, f"{name}.npy")
                    np.save(np_path, out.numpy())

                    name_path = os.path.join(path, f"{name}.txt")
                    with open(name_path, "w") as f:
                        for k, _ in enumerate(out):
                            patch_name = x_loader.dataset.files[k]
                            patch_name = patch_name.split("/")[6]
                            f.write(f"{patch_name}\n")

    def log(self, msg: str) -> None:
        self.args.logger.info(msg)


if __name__ == "__main__":
    parser = ArgumentParser("argument parser for MBC Camelyon extraction")

    parser.add_argument("--data-root", type=str,
                        default="/d1/dataset", help="dataset root")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="the set of GPUs to distribute the models to")
    parser.add_argument("--model", default="imagenet-trunc",
                        help="the model to use")

    args = parser.parse_args()
    args.logger = set_logger("INFO")

    args.logger.info("before dataset setup")
    for magnification in [20, 5]:
        args.magnification = magnification

        train = CAMELYON16Patches256Iterable(
            root=args.data_root,
            inner_workers=8,
            batch_size=64,
            split="train",
            magnification=magnification
        )
        test = CAMELYON16Patches256Iterable(
            root=args.data_root,
            inner_workers=8,
            batch_size=64,
            split="test",
            magnification=magnification
        )

        args.logger.info("making model")
        model: nn.Module
        if args.model == "imagenet":
            model = ResNet50Pretrained()
        elif args.model == "imagenet-trunc":
            model = resnet50_trunc_baseline(pretrained=True)
        elif args.model == "simclr":
            model = ResNetSimCLR(magnification=magnification)

        args.logger.info(
            f"starting: {magnification=} {len(train)=} {len(test)=}")
        trainer = CamelyonExtractor(args, model, train, test)
        mp.spawn(trainer.distributed_fit, nprocs=len(args.gpus), join=True)
