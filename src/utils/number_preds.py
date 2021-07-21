from typing import List

import numpy as np
import torch


def decode(pred: torch.tensor, alphabet: str) -> List[str]:
    pred = pred.permute(1, 0, 2).cpu().data().numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], alphabet))

    return outputs


def pred_to_string(pred: torch.tensor, alphabet: str) -> str:
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = "".join([alphabet[c] for c in out])

    return out
