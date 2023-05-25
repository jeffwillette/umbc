import csv
import os
from collections import namedtuple
from datetime import datetime
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.utils import (check_integrity,
                                        download_file_from_google_drive,
                                        verify_str_arg)
from torchvision.datasets.vision import VisionDataset

CSV = namedtuple("CSV", ["header", "index", "data"])


class CelebAResizerBase(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(CelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        self.filename = splits.index
        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor')
        self.attr_names = attr.header

        self.load = 0
        self.tx = 0

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        data, indices, headers = [], [], []

        fn = partial(os.path.join, self.root, self.base_folder)
        with open(fn(filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=' ', skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        if self.transform is not None:
            X = self.transform(X)
            X.save(os.path.join("img_align_celeba_32x32", self.filename[index]))

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return self.filename[index]

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


class CELEBAResizer(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        size=[32, 32],
        num_points=200,
        eval_mode='none',
        num_points_eval=200,
        download=False
    ):
        self.name = 'CELEBA'
        split = 'train' if train else 'test'
        self.size = size  # Original size: (218, 178)
        self.num_points = num_points
        self.eval_mode = eval_mode
        self.num_points_eval = num_points_eval

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(size),
                # transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.dataset = CelebAResizerBase(root=root, split=split, download=download, transform=transform)
        self.coordinates = torch.from_numpy(
            np.array([[int(i / size[1]) / size[0], (i % size[1]) / size[1]] for i in range(size[0] * size[1])])
        ).float()

        self.dataset_query = 0
        self.view = 0
        self.random_idx = 0
        self.ctx_xy = 0

    def __getitem__(self, index):
        image = self.dataset[index]
        return image

    # def __getitem__(self, index):
    #     start = datetime.now()
    #     image, _ = self.dataset[index]
    #     self.dataset_query += (datetime.now() - start).total_seconds()

    #     start = datetime.now()
    #     target_y = image.view(image.size(0), -1).transpose(0, 1)
    #     target_x = self.coordinates
    #     self.view += (datetime.now() - start).total_seconds()

    #     start = datetime.now()
    #     random_idx = torch.from_numpy(np.random.choice(np.product(self.size), size=self.num_points_eval if self.eval_mode == 'all' else self.num_points, replace=False))
    #     self.random_idx += (datetime.now() - start).total_seconds()

    #     start = datetime.now()
    #     context_x = torch.index_select(target_x, dim=0, index=random_idx)
    #     context_y = torch.index_select(target_y, dim=0, index=random_idx)
    #     self.ctx_xy += (datetime.now() - start).total_seconds()

    #     return context_x, context_y, target_x, target_y

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    celeba = CELEBAResizer(root='/d1/bruno/Datasets/CelebA', train=False, eval_mode='none', num_points_eval=1000)

    # for i in range(1000):
    #     context_x, context_y, target_x, target_y = celeba[i]
    print(len(celeba))
    print(celeba.dataset.target_type)
    for i, f in enumerate(celeba.dataset.filename):
        filename = celeba.dataset[i]
        assert f == filename
        print(filename)

    # print(context_x.size(), context_y.size())
    # print(target_x.size(), target_y.size())
