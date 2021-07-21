# Car plates recognition.
Solution to License Plate Recognition challenge (MADE Mail.ru) - https://www.kaggle.com/c/made-cv-2021-contest-02-license-plate-recognition
<br>
Leaderboard metric was mean Levenshtein distance, solution performed at test dataset with 0.88743 score.<br>
Solution contains MaskRCNN for bounding box (object detection task) and mask (segmentation task) prediction.<br>
Predicted bounding boxes and masks sends to typical CRNN for car plate string generation.<br>
Train stage consists of 3 tasks:
- Detector training.
- Prediction of bounding boxes and masks.
- RCNN training for OCR task.<br>

# Install requirements.
CUDA 11.1 must be installed on machine if GPU train will be used.<br>
Libs can be obtained via requirements.txt:<br>
```pip install -r requirements.txt```

# Configure train and predict processes.
"Configs" folder stores 4 yaml files, which describes main parameters for detector training and bounding boxes and masks prediction,
CRNN train process and prediction on test dataset.
- detector_train_params.yml - MaskRCNN training parameters.
- detector_predict_params.yml - prediction of bounding boxes and masks.
- ocr_train_params.yml - RCNN training parameters.
- ocr_predict_params.yml - prediction of car plates.<br>
Parameters can be changed via this config files edit.
  
# Train
```python main.py train```
Start train with default config files paths.

# Predict
```python main.py predict```
Make prediction with default config file path.