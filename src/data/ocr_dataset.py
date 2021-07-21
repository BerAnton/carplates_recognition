import os
from collections import Counter
from typing import Dict, List, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.utils.contour import quadrangle_rectangle_transform


class OCRDataset(Dataset):
    def __init__(self, marks: List, img_folder: str, alphabet: str, transforms: transforms.Compose = None):
        ocr_marks = []
        for items in marks:
            file_path = items["file"]
            for box in items["nums"]:

                ocr_marks.append(
                    {
                        "file": file_path,
                        "box": np.clip(box["box"], 0, None).tolist(),
                        "text": box["text"],
                        "boxed": False,
                    }
                )

                points = np.clip(box["box"], 0, None)
                x0, y0 = np.min(points[:, 0]), np.min(points[:, 1])
                x2, y2 = np.max(points[:, 0]), np.max(points[:, 1])

                ocr_marks.append(
                    {
                        "file": file_path,
                        "box": [x0, y0, x2, y2],
                        "text": box["text"],
                        "boxed": True,
                    }
                )

        self.marks = ocr_marks
        self.img_folder = img_folder
        self.transforms = transforms
        self.alphabet = alphabet

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.marks[idx]
        img_path = os.path.join(self.img_folder, item["file"])
        img = cv2.imread(img_path)

        if item["boxed"]:
            x_min, y_min, x_max, y_max = item["box"]
            img = img[y_min:y_max, x_min:x_max]
        else:
            points = np.clip(np.array(item["box"]), 0, None)
            img = quadrangle_rectangle_transform(img, points)

        text = item["text"]
        seq = [self.alphabet.find(char) + 1 for char in text]
        seq_len = len(seq)

        if self.transforms is not None:
            img = self.transforms(img)

        output = {"img": img, "text": text, "seq": seq, "seq_len": seq_len}

        return output

    def __len__(self) -> int:
        return len(self.marks)


def get_vocab_from_marks(marks: List):
    train_texts = []
    for item in marks:
        for num in item["nums"]:
            train_texts.append(num["text"])

    counts = Counter("".join(train_texts))
    alphabet = "".join(set("".join(train_texts)))
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    char_to_idx = {item[0]: idx + 1 for idx, item in enumerate(sorted_counts)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char, alphabet


def collate_fn_ocr(batch):
    """Function for torch.utils.data.DataLoader for batch collecting.
    Accepts list of dataset __get_item__ return values (dicts).
    Returns dict with same keys but values are either torch.Tensors of batched images, sequences, and so.
    """
    images, seqs, seq_lens, texts = [], [], [], []
    for sample in batch:
        images.append(sample["img"])
        seqs.extend(sample["seq"])
        seq_lens.append(sample["seq_len"])
        texts.append(sample["text"])
    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()
    batch = {"image": images, "seq": seqs, "seq_len": seq_lens, "text": texts}
    return batch
