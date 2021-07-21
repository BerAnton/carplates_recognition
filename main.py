"""Main module for running train and prediction pipelines"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import logging

import torch

from src.train_pipeline import train_pipeline
from src.predict_pipeline import predict_pipeline

logger = logging.getLogger(__name__)

DEFAULT_DETECTOR_TRAIN_CONFIG_PATH = "configs/detector_train_params.yml"
DEFAULT_DETECTOR_PREDICT_CONFIG_PATH = "configs/detector_predict_params.yml"
DEFAULT_OCR_TRAIN_CONFIG_PATH = "configs/ocr_train_params.yml"
DEFAULT_OCR_PREDICT_CONFIG_PATH = "configs/ocr_predict_params.yml"


def callback_train(arguments: Namespace) -> None:
    """callback for train model"""
    train_pipeline(
        arguments.detector_train_config_path, arguments.detector_predict_config_path, arguments.ocr_train_config_path
    )


def callback_predict(arguments: Namespace) -> None:
    """callback for make prediction"""
    predict_pipeline(arguments.ocr_predict_config_path)


def setup_parser(parser: ArgumentParser) -> None:
    """Setup CLI-parser"""
    subparsers = parser.add_subparsers(help="choose mode: train or predict")
    train_parser = subparsers.add_parser(
        "train",
        help="train model for facial landmarks recognition",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    predict_parser = subparsers.add_parser(
        "predict",
        help="predict landmarks on given dataset",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    train_parser.set_defaults(callback=callback_train)
    predict_parser.set_defaults(callback=callback_predict)

    train_parser.add_argument(
        "--detector-train-config-path",
        default=DEFAULT_DETECTOR_TRAIN_CONFIG_PATH,
        help="path to detector train config, default path is %(default)s",
    )
    train_parser.add_argument(
        "--detector-predict-config-path",
        default=DEFAULT_DETECTOR_PREDICT_CONFIG_PATH,
        help="path to detector predict config, default path is %(default)s",
    )
    train_parser.add_argument(
        "--ocr-train-config-path",
        default=DEFAULT_OCR_TRAIN_CONFIG_PATH,
        help="path to ocr train config, default path is %(default)s",
    )
    predict_parser.add_argument(
        "--ocr-predict-config-path",
        default=DEFAULT_OCR_PREDICT_CONFIG_PATH,
        help="path to ocr predict config, default path is %(default)s",
    )


def main() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = ArgumentParser(
        prog="car plates photo recognition",
        description="tool for train CNN for facial landmarks recognition",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
