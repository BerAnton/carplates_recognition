import os
import yaml
import json
import logging
from pathlib import Path
import glob
from PIL import Image

import tqdm
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer

from src.model import get_detector_model, CRNN
from src.data import DetectionDataset, OCRDataset, get_vocab_from_marks, collate_fn, collate_fn_ocr
from src.utils import simplify_contour, load_json, npEncoder
from src.transforms import Resize


def train_detector(model, loader: DataLoader, optimizer: Optimizer, scheduler, device: torch.device):
    """Train loop for model"""
    model.train()
    train_loss = []
    for i, (images, targets) in tqdm.tqdm(enumerate(loader), leave=True, position=0, total=len(loader)):

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss.append(losses.item())
        if (i + 1) % 200 == 0:
            mean_loss = np.mean(train_loss)
            print(f"Loss: {mean_loss:.5f}")
            scheduler.step(mean_loss)
            train_loss = []


def predict_bbox_masks(model, test_transforms, data_path, threshold_score, threshold_mask, device):
    test_images = glob.glob(os.path.join(data_path, "test/*"))
    preds = []
    model.eval()
    approx = None

    for file in tqdm.tqdm(test_images, position=0, leave=False):

        img = Image.open(file).convert("RGB")
        img_tensor = test_transforms(img)
        with torch.no_grad():
            predictions = model(img_tensor.to(device))
        prediction = predictions[0]

        pred = dict()
        pred["file"] = file
        pred["nums"] = []

        for i in range(len(prediction["boxes"])):
            x_min, y_min, x_max, y_max = map(int, prediction["boxes"][i].tolist())
            label = int(prediction["labels"][i].cpu())
            score = float(prediction["scores"][i].cpu())
            mask = prediction["masks"][i][0, :, :].cpu().numpy()

            if score > threshold_score:
                contours, _ = cv2.findContours((mask > threshold_mask).astype(np.uint8), 1, 1)
                approx = simplify_contour(contours[0], n_corners=4)
            if approx is None:
                x0, y0 = x_min, y_min
                x1, y1 = x_max, y_min
                x2, y2 = x_min, y_max
                x3, y3 = x_max, y_max
            else:
                x0, y0 = approx[0][0][0], approx[0][0][1]
                x1, y1 = approx[1][0][0], approx[1][0][1]
                x2, y2 = approx[2][0][0], approx[2][0][1]
                x3, y3 = approx[3][0][0], approx[3][0][1]

            points = [[x0, y0], [x2, y2], [x1, y1], [x3, y3]]
            pred["nums"].append(
                {
                    "box": points,
                    "bbox": [x_min, y_min, x_max, y_max],
                }
            )
        preds.append(pred)

    return preds


def crnn_train(model, loader, optimizer, scheduler, epochs: int, device: torch.device) -> None:
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        print_loss = []

        for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), leave=False, position=0)):
            images = batch["image"].to(device)
            seqs_gt = batch["seq"]
            seq_lens_gt = batch["seq_len"]

            seqs_pred = model(images).cpu()
            log_probs = F.log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size[1]).int()

            loss = F.ctc_loss(
                log_probs=log_probs,  # (T, N, C)
                targets=seqs_gt,  # N, S or sum(target_lengths)
                input_lengths=seq_lens_pred,  # N
                target_lengths=seq_lens_gt,  # N
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print_loss.append(loss.item())
            if (i + 1) % 20 == 0:
                mean_loss = np.mean(print_loss)
                print(f"Loss: {mean_loss:.5f}")
                scheduler.step(mean_loss)
                print_loss = []

            epoch_losses.append(loss.item())

        print(f"Epoch: {epoch + 1}, Loss: {np.meam(epoch_losses)}")


def train_pipeline(
    train_detector_config_path: Path, predict_detector_config_path: Path, ocr_train_config_path: Path
) -> None:
    """Train pipeline for landmarks recognition.
    :args:
         - train_detector_config_path - path to config with train params."""
    with open(train_detector_config_path, "r") as fin:
        detector_train_config = yaml.safe_load(fin)

    train_input_data_path = detector_train_config["input_data_path"]
    model_save_path = detector_train_config["model_save_path"]
    detector_model_name = detector_train_config["model_name"]
    train_size = detector_train_config["train_dataset_size"]
    detector_batch_size = detector_train_config["detector_batch_size"]
    detector_epochs = detector_train_config["epochs"]
    detector_lr = detector_train_config["lr"]
    cuda = detector_train_config["use_cuda"]

    # split marks into train and val datasets
    marks = load_json(os.path.join(train_input_data_path, "train.json"))
    test_start = int(train_size * len(marks))
    train_marks = marks[:test_start]
    val_marks = marks[test_start:]

    device = torch.device("cuda") if cuda else torch.device("cpu")
    train_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = DetectionDataset(marks=train_marks, img_folder=train_input_data_path, transforms=train_transforms)
    val_dataset = DetectionDataset(marks=val_marks, img_folder=train_input_data_path, transforms=train_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=detector_batch_size,
        drop_last=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=detector_batch_size,
        drop_last=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    """Step 1. Train and save detector model."""
    detector_model = get_detector_model()
    detector_model.to(device)
    optimizer = AdamW(detector_model.parameters(), lr=detector_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)
    for epoch in range(detector_epochs):
        train_detector(detector_model, train_loader, optimizer, scheduler, device)
    torch.save(detector_model.state_dict(), os.path.join(model_save_path, detector_model_name))  # save detector model

    """Step 2. Generate bounding boxes and masks with trained detector model."""
    with open(predict_detector_config_path, "r") as fin:
        detector_predict_config = yaml.safe_load(fin)

    input_data_path = detector_predict_config["input_data_path"]
    threshold_score = detector_predict_config["threshold_score"]
    threshold_mask = detector_predict_config["threshold_mask"]
    preds_save_path = detector_predict_config["preds_save_path"]
    preds_name = detector_predict_config["preds_name"]

    bbox_masks_preds = predict_bbox_masks(
        detector_model, train_transforms, input_data_path, threshold_score, threshold_mask, device
    )

    with open(os.path.join(preds_save_path, preds_name), "w") as fout:
        json.dump(bbox_masks_preds, fout, cls=npEncoder)  # serialize detector predictions

    """Step 3. OCR on predicted bounding boxes and masks.
       Data input path are same with detector."""
    with open(ocr_train_config_path, "r") as fin:
        ocr_train_config = yaml.safe_load(fin)

    ocr_model_name = ocr_train_config["model_name"]
    ocr_batch_size = ocr_train_config["batch_size"]
    ocr_lr = ocr_train_config["lr"]
    ocr_epochs = ocr_train_config["epochs"]
    alphabet_save_path = ocr_train_config["alphabet_save_path"]

    ocr_transforms = transforms.Compose(
        [
            Resize(size=(320, 64)),
            transforms.ToTensor(),
        ]
    )

    char_to_idx, idx_to_char, alphabet = get_vocab_from_marks(train_marks)
    with open(alphabet_save_path, "w") as fout:
        fout.write(alphabet)

    train_ocr_dataset = OCRDataset(
        marks=train_marks, img_folder=train_input_data_path, alphabet=alphabet, transforms=ocr_transforms
    )
    val_ocr_dataset = OCRDataset(
        marks=val_marks, img_folder=train_input_data_path, alphabet=alphabet, transforms=ocr_transforms
    )

    train_ocr_loader = DataLoader(
        train_ocr_dataset,
        batch_size=ocr_batch_size,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn_ocr,
        timeout=0,
        shuffle=True,
    )
    val_ocr_loader = DataLoader(
        val_ocr_dataset,
        batch_size=ocr_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn_ocr,
        timeout=0,
    )

    crnn = CRNN(alphabet)
    crnn.to(device)
    optimizer = torch.optim.Adam(crnn.parameters(), lr=ocr_lr, amsgrad=True, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    crnn_train(crnn, train_ocr_loader, optimizer, scheduler, ocr_epochs, device)
    torch.save(crnn.state_dict(), os.path.join(model_save_path, ocr_model_name))  # save ocr model
