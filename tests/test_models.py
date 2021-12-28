"""
File with tests of models
"""
import os, sys
import logging

import pytest

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn

from models import LSTModel, GRUModel
from data_utilities import read_text, text_to_seq, train, evaluate
from constants import *

EXAMPLE_CORPUS_PATH = '/Users/photosartd/PycharmProjects/ChaRNN/example_corpuses/max_poetry.txt'


@pytest.mark.slow
def test_lstm_rnn(capsys, caplog):
    text = read_text(EXAMPLE_CORPUS_PATH)
    sequence, token_to_idx, idx_to_token = text_to_seq(text)
    model = LSTModel(len(token_to_idx),
                     DEFAULT_HIDDEN_SIZE,
                     DEFAULT_EMBEDDING_SIZE
                     )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=OPTIMIZER_LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        verbose=True,
        factor=0.5
    )
    with caplog.at_level(logging.DEBUG):
        train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            sequence=sequence,
            idx_to_token=idx_to_token,
            token_to_idx=token_to_idx,
            batch_size=DEFAULT_BATCH_SIZE,
            seq_len=DEFAULT_SEQ_LEN,
            epochs=102,
            device=DEFAULT_DEVICE,
            scheduler=scheduler
        )
        captured = capsys.readouterr()


def test_gru_rnn(capsys, caplog):
    text = read_text(EXAMPLE_CORPUS_PATH)
    sequence, token_to_idx, idx_to_token = text_to_seq(text)
    model = GRUModel(len(token_to_idx),
                     DEFAULT_HIDDEN_SIZE,
                     DEFAULT_EMBEDDING_SIZE
                     )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=OPTIMIZER_LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        verbose=True,
        factor=0.5
    )
    with caplog.at_level(logging.DEBUG):
        train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            sequence=sequence,
            idx_to_token=idx_to_token,
            token_to_idx=token_to_idx,
            batch_size=DEFAULT_BATCH_SIZE,
            seq_len=DEFAULT_SEQ_LEN,
            epochs=102,
            device=DEFAULT_DEVICE,
            scheduler=scheduler
        )
        captured = capsys.readouterr()
