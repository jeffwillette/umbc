from argparse import Namespace
from typing import Any, List, Tuple

import torch
from torch.utils.data import DataLoader

from data.celeba import CELEBAResized
from data.cmu_book import CMUBookSummaryDataset
from data.eurlex import EurLexDataset
from data.toy_classification import MixtureOfGaussians

T = torch.Tensor


def list_collate(batch: Any) -> Tuple[Any, T, Any]:
    """batch: List[Tuple[DataLoaderReturn]]"""
    x_out = [t[0] for t in batch]
    y_out = [t[1] for t in batch]
    i_out = [t[2] for t in batch]

    return x_out, torch.tensor(y_out), i_out


def cat_mask(a: T, b: int) -> T:
    return torch.cat((a, torch.zeros(b)))


def merge(inputs: List[T], masks: List[T]) -> Tuple[T, T]:
    lengths = [v.size(0) for v in inputs]
    diffs = [max(lengths) - v.size(0) for v in inputs]
    dims = [v.size(-1) for v in inputs]
    dim = dims[0]

    # assert all inputs are the same dimension
    assert all([v == dim for v in dims])

    padded_x = [torch.cat((v, torch.zeros(d, dim)), dim=0)
                for (v, d) in zip(inputs, diffs)]
    padded_masks = [cat_mask(m, d)
                    for (m, d) in zip(masks, diffs)]

    return torch.stack(padded_x), torch.stack(padded_masks)


def pad_collate(batch: Any) -> Tuple[T, T, T]:
    x_20, mask_20, y = zip(*batch)

    inputs_20, masks_20 = merge(x_20, mask_20)
    labels = torch.tensor(y)

    return inputs_20, masks_20, labels


def get_mixture_of_gaussians(args: Namespace) -> Tuple[DataLoader, ...]:
    train = MixtureOfGaussians(dim=args.dim, mvn_type=args.mvn_type)
    loader = DataLoader(train, batch_size=1, shuffle=True)
    return loader, loader, loader


def get_celeba(args: Namespace) -> Tuple[DataLoader, ...]:
    train = CELEBAResized(root=args.data_root, split="train",
                          num_points=args.train_set_size,
                          num_points_eval=args.train_set_size)

    val = CELEBAResized(root=args.data_root, split="valid",
                        num_points=args.test_set_size,
                        num_points_eval=args.test_set_size,
                        eval_mode="all")

    test = CELEBAResized(root=args.data_root, split="test",
                         num_points=args.test_set_size,
                         num_points_eval=args.test_set_size,
                         eval_mode="all")
    trainloader = DataLoader(train, shuffle=True, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val, shuffle=False, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)
    testloader = DataLoader(test, shuffle=False, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)
    return trainloader, val_loader, testloader


def get_cmu_book_summary(args: Namespace) -> Tuple[DataLoader, ...]:
    train = CMUBookSummaryDataset(root=args.data_root, split="train",
                                  num_points=args.train_set_size, shuffle=True)
    val = CMUBookSummaryDataset(root=args.data_root, split="dev",
                                num_points=args.test_set_size, shuffle=False)
    test = CMUBookSummaryDataset(root=args.data_root, split="test",
                                 num_points=args.test_set_size, shuffle=False)

    def collate_fn(data):
        def merge(sequences):
            lengths = [seq.size(0) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs

        input_ids, input_mask, labels = zip(*data)

        input_ids = merge(input_ids)
        input_mask = merge(input_mask)
        labels = torch.stack(labels, dim=0)

        return input_ids, input_mask, labels

    trainloader = DataLoader(train, collate_fn=collate_fn,
                             shuffle=True, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val, collate_fn=collate_fn,
                            shuffle=False, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test, collate_fn=collate_fn,
                             shuffle=False, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True)

    return trainloader, val_loader, test_loader


def get_eurlex(args: Namespace) -> Tuple[DataLoader, ...]:
    train = EurLexDataset(
        args.data_root, split="train", num_points=args.train_set_size, shuffle=True
    )
    val = EurLexDataset(
        args.data_root, split="dev", num_points=args.test_set_size, shuffle=False
    )
    test = EurLexDataset(
        args.data_root, split="test", num_points=args.test_set_size, shuffle=False
    )

    def collate_fn(data):
        def merge(sequences):
            lengths = [seq.size(0) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs

        input_ids, input_mask, labels = zip(*data)

        input_ids = merge(input_ids)
        input_mask = merge(input_mask)
        labels = torch.stack(labels, dim=0)

        return input_ids, input_mask, labels

    trainloader = DataLoader(train, collate_fn=collate_fn,
                             shuffle=True, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val, collate_fn=collate_fn,
                            shuffle=False, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test, collate_fn=collate_fn,
                             shuffle=False, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True)

    return trainloader, val_loader, test_loader


deref = {
    "toy-mixture-of-gaussians": get_mixture_of_gaussians,
    "cmu_book_summaries": get_cmu_book_summary,
    "celeba": get_celeba,
    "eurlex": get_eurlex,
}


def get_dataset(args: Namespace, **kwargs: Any) -> Tuple[DataLoader, ...]:
    if args.dataset not in deref.keys():
        raise NotImplementedError(
            f"dataset: {args.dataset} is not implemented")

    return deref[args.dataset](args, **kwargs)  # type: ignore


def get_dataset_by_name(
    args: Namespace,
    name: str = "",
    **kwargs: Any
) -> Tuple[DataLoader, ...]:
    if name not in deref.keys():
        raise NotImplementedError(f"dataset: {name} is not implemented")

    return deref[name](args, **kwargs)  # type: ignore
