"""
File with data_utilities.py tests
"""
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch

from data_utilities import read_text, text_to_seq, get_batch


EXAMPLE_CORPUS_PATH = '/Users/photosartd/PycharmProjects/ChaRNN/example_corpuses/max_poetry.txt'


def test_read_text():
    text = read_text(EXAMPLE_CORPUS_PATH)
    assert all(not letter.isupper() for letter in text)


def test_text_to_seq():
    text = read_text(EXAMPLE_CORPUS_PATH)
    sequence, token_to_idx, idx_to_token = text_to_seq(text)
    assert len(text) == len(sequence)
    assert all(idx in sequence for idx in idx_to_token.keys())


def test_get_batch():
    text = read_text(EXAMPLE_CORPUS_PATH)
    sequence, token_to_idx, idx_to_token = text_to_seq(text)
    batch_size = 16
    seq_len = 256
    data, targets = get_batch(sequence, batch_size, seq_len)
    assert data.shape == torch.Size((batch_size, seq_len - 1, 1))
    assert targets.shape == torch.Size((batch_size, seq_len - 1, 1))

