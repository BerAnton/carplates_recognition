from typing import Tuple
from PIL import Image

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision.transforms import transforms

from contour import simplify_contour, quadrangle_rectangle_transform


def visualize_predictions(
    file: str,
    model,
    transforms: transforms.Compose,
    device: torch.device,
    verbose: bool = True,
    threshold: float = 0.0,
    n_colors=None,
    id_to_name=None,
):
    img = Image.open(file)
    img_tensor = transforms(img)

    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model([img_tensor.to(device)])
    prediction = predictions[0]

    if n_colors is None:
        n_colors = model.roi_heads.box_predictor.cls_score.out_features

    palette = sns.color_palette(None, n_colors)

    img = cv2.imread(file, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    image = img

    black_img = np.zeros(image.shape, image.dtype)
    black_img[:, :] = (0, 0, 0)
    for i in range(len(prediction["boxes"])):
        x_min, y_min, x_max, y_max = map(int, prediction["boxes"][i].tolist())
        label = int(prediction["labels"][i].cpu())
        score = float(prediction["scores"][i].cpu())
        mask = prediction["masks"][i][0, :, :].cpu().numpy()
        name = id_to_name[label]
        color = palette[label]

        if verbose:
            if score > threshold:
                print(f"Class: {name}, Confidence: {score}")
        if score > threshold:
            crop_img = image[y_min:y_max, x_min:x_max]
            print("Bounding box:")
            show_image(crop_img, figsize=(10, 2))

            contours, _ = cv2.findContours((mask > 0.05).astype(np.uint8), 1, 1)
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

            pts = np.array([[x0, y0], [x2, y2], [x1, y1], [x3, y3]])
            crop_mask_img = quadrangle_rectangle_transform(img, pts)
            print("Rotated img:")
            crop_mask_img = cv2.resize(crop_mask_img, (320, 64), interpolation=cv2.INTER_AREA)
            show_image(crop_mask_img, figsize=(10, 2))
            if approx is not None:
                cv2.drawContours(image, [approx], 0, (255, 0, 255), 3)
            image = cv2.circle(image, (x0, y0), radius=5, color=(0, 0, 255), thickness=-1)
            image = cv2.circle(image, (x1, y1), radius=5, color=(0, 0, 255), thickness=-1)
            image = cv2.circle(image, (x2, y2), radius=5, color=(0, 0, 255), thickness=-1)
            image = cv2.circle(image, (x3, y3), radius=5, color=(0, 0, 255), thickness=-1)

            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), np.array(color) * 255, 2)

    show_image(image)

    return prediction


def show_image(image: Image, figsize: Tuple = (16, 9), reverse: bool = True):
    plt.figure(figsize=figsize)
    if reverse:
        plt.imshow(image[..., ::-1])
    else:
        plt.imshow(image)
    plt.axis("off")
    plt.show()
