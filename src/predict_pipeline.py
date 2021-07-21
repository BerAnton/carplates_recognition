from pathlib import Path

import pandas as pd
import yaml

import tqdm
import numpy as np
import cv2
import torch
from torchvision.transforms import transforms

from src.model import CRNN
from src.transforms import Resize
from src.utils import load_json, quadrangle_rectangle_transform, decode


def predict_pipeline(ocr_predict_config_path: Path) -> None:
    with open(ocr_predict_config_path, "r") as fin:
        ocr_predict_config = yaml.safe_load(fin)

    data_path = ocr_predict_config["bbox_masks_path"]
    model_path = ocr_predict_config["ocr_model_path"]
    alphabet_path = ocr_predict_config["alphabet_path"]
    result_path = ocr_predict_config["result_df_path"]
    cuda = ocr_predict_config["use_cuda"]

    device = torch.device("cuda") if cuda else torch.device("cpu")

    test_transforms = transforms.Compose([transforms.ToTensor()])

    with open(alphabet_path, "r") as fin:
        alphabet = fin.read()

    model = CRNN(alphabet)
    with open(model_path, "rb") as fin:
        state_dict = torch.load(fin, map_location=device)
        model.load_state_dict(state_dict)

    test_marks = load_json(data_path)
    model.eval()
    resizer = Resize()

    filename_result = []
    plates_result = []

    for item in tqdm.tqdm(test_marks, leave=False, position=0):

        img_path = item["file"]
        img = cv2.imread(img_path)

        temp_results = []
        for box in item["nums"]:
            x_min, y_min, x_max, y_max = box["bbox"]
            img_bbox = resizer(img[y_min:y_max, x_min:x_max])
            img_bbox = test_transforms(img_bbox)
            img_bbox = img_bbox.unsqueeze(0)

            points = np.clip(np.array(box["box"]), 0, None)
            img_polygon = resizer(quadrangle_rectangle_transform(img, points))
            img_polygon = test_transforms(img_polygon)
            img_polygon = img_polygon.unsqueeze(0)

            preds_bbox = model(img_bbox.to(device)).cpu().detach()
            preds_poly = model(img_polygon.to(device)).cpu().detach()

            preds = preds_poly + preds_bbox
            num_text = decode(preds, alphabet)[0]
            temp_results.append((x_min, num_text))

        results = sorted(temp_results, key=lambda x: x[0])
        num_list = [x[1] for x in results]

        plates_string = " ".join(num_list)
        file_name = img_path[img_path.find("test/") :]

        filename_result.append(file_name)
        plates_result.append(plates_string)

        result_df = pd.DataFrame({"file_name": filename_result, "Plate": plates_result})
        result_df.to_csv(result_path, index=False)
