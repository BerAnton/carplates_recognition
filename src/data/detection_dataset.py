import os
from typing import List, Dict, Tuple
from PIL import Image

import numpy as np
from matplotlib.path import Path
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset


# noinspection PyTypeChecker
class DetectionDataset(Dataset):
    def __init__(self, marks: List, img_folder: str, transforms: transforms.Compose = None) -> None:

        self.marks = marks
        self.img_folder = img_folder
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        item = self.marks[idx]
        img_path = os.path.join(self.img_folder, item["file"])

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        box_coords = item["nums"]
        boxes = []
        labels = []
        masks = []
        for box in box_coords:
            points = np.array(box["box"])
            x0, y0 = np.min(points[:, 0]), np.min(points[:, 1])
            x2, y2 = np.max(points[:, 0]), np.max(points[:, 1])
            boxes.append([x0, y0, x2, y2])
            labels.append(1)

            nx, ny = w, h
            poly_verts = points
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            path = Path(poly_verts)
            grid = path.contains_points(points)
            grid = grid.reshape((ny, nx)).astype(int)
            masks.append(grid)

        boxes = torch.as_tensor(boxes)
        labels = torch.as_tensor(labels)
        masks = torch.as_tensor(masks)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self) -> int:
        return len(self.marks)


def collate_fn(batch):
    return tuple(zip(*batch))
