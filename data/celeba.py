import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
import PIL  # type: ignore
import torch
import torchvision  # type: ignore
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class CelebAResizedBase(torchvision.datasets.CelebA):
    """
    This dataset assumes one has access to the already resized celeba images in the same
    directory as the orignial celeba datasets in a subdirectory names 'img_align_celeba_32x32'
    """

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(
            self.root, self.base_folder, "img_align_celeba_32x32", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)


class CELEBAResized(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        size: Tuple[int, int] = (32, 32),
        num_points: int = 200,
        eval_mode: str = 'none',
        num_points_eval: int = 200,
        download: bool = False,
    ):
        self.name = 'CELEBA'
        self.split = split

        self.size = size  # Original size: (218, 178)
        self.num_points = num_points
        self.eval_mode = eval_mode
        self.num_points_eval = num_points_eval

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.dataset = CelebAResizedBase(
            root=root, split=split, download=download, transform=transform)
        if split == "train":
            self.dataset = [
                self.dataset[i] for i in tqdm(range(len(self.dataset)), leave=False, desc="load celeba")]
        self.coordinates = torch.from_numpy(
            np.array([[int(i / size[1]) / size[0], (i % size[1]) / size[1]]
                     for i in range(size[0] * size[1])])
        ).float()

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        target_y = image.view(image.size(0), -1).transpose(0, 1)
        target_x = self.coordinates

        random_idx = torch.from_numpy(
            np.random.choice(
                np.product(self.size),
                size=self.num_points_eval if self.eval_mode == 'all' else self.num_points, replace=False)
        )
        context_x = torch.index_select(target_x, dim=0, index=random_idx)
        context_y = torch.index_select(target_y, dim=0, index=random_idx)
        context = torch.cat([context_x, context_y], dim=-1)

        return context, target_x, target_y

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    celeba = CELEBAResized(root='/d1/dataset', split="test",
                           eval_mode='none', num_points_eval=1000)
    context, target_x, target_y = celeba[0]
    print(len(celeba))
    print(context.size())

    print(target_x.size(), target_y.size())
