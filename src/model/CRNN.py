import torch.nn as nn

from src.model.sequence_predictor import SequencePredictor
from src.model.feature_extractor import FeatureExtractor


class CRNN(nn.Module):
    def __init__(
        self,
        alphabet,
        cnn_input_size=(64, 320),
        cnn_output_len=20,
        rnn_hidden_size=128,
        rnn_num_layers=2,
        rnn_dropout=0.3,
        rnn_bidirectional=False,
    ):
        super(CRNN, self).__init__()
        self.alphabet = alphabet

        self.features_extractor = FeatureExtractor(input_size=cnn_input_size, output_len=cnn_output_len)

        self.sequence_predictor = SequencePredictor(
            input_size=self.features_extractor.num_output_features,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            num_classes=(len(alphabet) + 1),
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
        )

    def forward(self, x):
        features = self.features_extractor(x)
        sequence = self.sequence_predictor(features)
        return sequence
