import logging
import os
import random
from hashlib import md5
from multiprocessing import Pool
from typing import Any, Dict, Iterator, List, Tuple

import h5py  # type: ignore
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms  # type: ignore
from utils import set_logger

T = torch.Tensor

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.7164, 0.5811, 0.7227), (0.2507, 0.3040, 0.2219))
])


def open_image(path: str) -> T:
    return transform(Image.open(path))  # type: ignore


resize = transforms.Resize(64)


def open_image_npy(path: str) -> NDArray:
    return np.array(resize(Image.open(path)))  # type: ignore


def cond(f: str, split: str) -> bool:
    if split == "train":
        return "test_" not in f
    else:
        return "test_" in f


def get_labels(
    split: str,
    files: List[str],
    test_label_file_path: str
) -> List[int]:
    if split in ["train", "val"]:
        return [1 if "tumor" in v else 0 for v in files]

    file_labels = {}
    with open(test_label_file_path, "r") as f:
        for line in f:
            n, y = line.split(" ")
            file_labels[n] = int(y)

    return [file_labels[k] for k in files]


MAGNIFICATIONS = {20: "10.0", 5: "30.0"}


def get_tensors_h5(args: Any) -> T:
    h5path, i = args
    print(f"loading: {i}")
    with h5py.File(h5path, "r") as f:
        return torch.stack([
            torch.from_numpy(np.array(f[k])) for k in f.keys()
        ])


def get_tensors(args: Any) -> T:
    np_path, i = args
    return torch.from_numpy(np.load(np_path))


class CAMELYON16Patches256Extracted(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
    ) -> None:
        super().__init__()

        self.name = "CAMELYON16Patches256"
        self.split = split

        self.twenty = os.path.join(
            root, "CAMELYON16-patches-256", "10.0-numpy-simclr")

        self.test_label_file = os.path.join(
            root, "CAMELYON16-patches-256", "test-labels.txt")

        twenty_files = sorted(os.listdir(self.twenty))
        twenty_files = [v for v in twenty_files if ".npy" in v]

        self.twenty_files: List[str] = []
        for f in twenty_files:
            if cond(f.split(".")[0], split):
                self.twenty_files.append(os.path.join(self.twenty, f))

        filenames = [f.split("/")[-1].split(".")[0]
                     for f in self.twenty_files]
        self.targets = get_labels(
            split, filenames, self.test_label_file)

        args = [(f, i) for (i, f) in enumerate(self.twenty_files)]
        self.data_twenty = list(map(get_tensors, args))

    def __len__(self) -> int:
        return len(self.twenty_files)

    def __getitem__(self, i: int) -> Tuple[T, T, int]:
        twenty = self.data_twenty[i]

        if "train" in self.split and twenty.size(0) > 10000:
            idx = torch.randperm(twenty.size(0))[:10000]
            twenty = twenty[idx]

        return (twenty, torch.ones(twenty.size(0)), self.targets[i])


class CAMELYON16Patches256Iterable(Dataset):
    def __init__(
        self,
        root: str,
        inner_workers: int,
        batch_size: int,
        split: str = "train",
        magnification: int = 20,
        p: float = 0.0,
    ) -> None:
        super().__init__()
        self.name = "CAMELYON16Patches256"
        self.split = split
        self.workers = inner_workers
        self.batch_size = batch_size
        self.p = p

        self.root = os.path.join(
            root, "CAMELYON16-patches-256", MAGNIFICATIONS[magnification])

        self.test_label_file = os.path.join(
            root, "CAMELYON16-patches-256", "test-labels.txt")

        files = sorted(os.listdir(self.root))
        self.files: Dict[str, List[str]] = {
            v: [] for v in files if cond(v, split)}

        for k in self.files.keys():
            filenames = os.listdir(os.path.join(self.root, k))
            self.files[k] = filenames

        self.targets = get_labels(split, list(
            self.files.keys()), self.test_label_file)

    def calculate_mu(self, logger: logging.Logger) -> T:
        s, n = torch.zeros(3), 0
        for i in range(len(self)):
            logger.info(f"{i}, {s=}, {n=}, {s/n=}")
            x_ds, _, _ = self[i]
            for x in x_ds:
                # the end dims will always be the same so we can
                # take a mean over them safely
                x = x.mean(dim=(2, 3))
                s += x.sum(dim=0)
                n += x.size(0)

        logger.info(f"mu: {s / n}")
        return s / n

    def calculate_sigma(self, mu: T, logger: logging.Logger) -> T:
        s, n = torch.zeros(3), 0
        mu = mu.view(1, -1, 1, 1)

        for i in range(len(self)):
            logger.info(f"{i}, {s=}, {n=}, {torch.sqrt(s / n)=}")
            x_ds, _, _ = self[i]
            for x in x_ds:
                # the end dims will always be the same so we can
                # take a mean over them safely
                x = ((x - mu) ** 2).mean(dim=(2, 3))
                s += x.sum(dim=0)
                n += x.size(0)

        logger.info(f"sigma: {torch.sqrt(s / n)}")
        return torch.sqrt(s / n)

    def __len__(self) -> int:
        return len(self.files.keys())

    def __getitem__(self, i: int) -> Tuple[DataLoader, int, int]:
        key = list(self.files.keys())[i]
        filenames = self.files[key]
        filepaths = [os.path.join(self.root, key, v) for v in filenames]

        if self.split != "test":
            random.shuffle(filepaths)
            if self.p > 0.0:
                drop_mask = (torch.rand(len(filepaths)) > self.p).tolist()
                drop_mask = [i for i, v in enumerate(drop_mask) if v]
                filepaths = [filepaths[i] for i in drop_mask]

        dl = DataLoader(
            SetIterator(filepaths),
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False
        )

        return dl, self.targets[i], i  # type: ignore


class SetIterator(IterableDataset):
    def __init__(self, files: List[str]) -> None:
        self.files = files

    def single_worker(self) -> Iterator[T]:
        return iter(open_image(v) for v in self.files)

    def get_worker_files(self, n_workers: int, worker_id: int) -> List[str]:
        """
        lets each worker seqeuntially work on the next set element so that
        the work of loading each chunk is evenly distributed among workers.
        """
        worker_files, i = [], 0
        while True:
            idx = i * n_workers + worker_id
            if idx >= len(self.files):
                break

            worker_files.append(self.files[idx])
            i += 1
        return worker_files

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[T]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process, return full iterator
            return self.single_worker()

        # in a worker process, split workload
        worker_files = self.get_worker_files(
            worker_info.num_workers, worker_info.id)

        return iter(open_image(v) for v in worker_files)


class CAMELYON16(Dataset):
    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = os.path.join(root, "CAMELYON16")
        self.md5 = "checksums.md5"

    def get_test_labels(self) -> None:
        # if there is a test file in the annotation directory, that means
        # it is a positive sample all other test files are from
        # the negative class

        annot = [v for v in os.listdir(os.path.join(
            self.root, "annotations")) if "test_" in v]
        annot = [v.split(".")[0] for v in annot]

        test_files = [v for v in os.listdir(
            os.path.join(self.root, "images")) if "test_" in v]
        test_files = [v.split(".")[0] for v in test_files]

        test_labels = [0 for _ in test_files]

        for i, f in enumerate(test_files):
            if f in annot:
                test_labels[i] = 1

        for f, l in zip(test_files, test_labels):
            print(f, l)

    def check_integrity(self) -> None:
        with open(os.path.join(self.root, self.md5)) as f:
            lines = f.read().splitlines()

        # The readme is expected to be corrupted because we added an appendage
        # to the beginning. the rest of the files pass the integrity check on
        # /d1/dataset
        for line in lines:
            checksum, path = line.split(" ")
            path = os.path.join(self.root, path[1:])

            with open(path, "rb") as hf:
                actual = md5(hf.read()).hexdigest()
                if checksum != actual:
                    print(f"\nfile is corrupted: {path}\n")
                    continue

            print(".", end="", flush=True)


def calculate_mu_sigma(logger: logging.Logger) -> None:
    logger.info("loading dataset")
    camelyon16 = CAMELYON16Patches256Iterable(
        "/d1/dataset", inner_workers=8, batch_size=128, split="train")

    # mu = camelyon16.calculate_mu(logger)  # [0.7164, 0.5811, 0.7227]
    sigma = camelyon16.calculate_sigma(
        torch.tensor([0.7164, 0.5811, 0.7227]),
        logger
    )
    print(sigma)
