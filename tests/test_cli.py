"""
Tests if CLI works as expected
"""
import os, sys
import logging
from argparse import ArgumentParser

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from charnn import setup_parser
from constants import *
from test_data_utilities import EXAMPLE_CORPUS_PATH


def test_train(capsys, caplog):
    parser = ArgumentParser()
    setup_parser(parser)
    command = f'{TRAIN} {MODEL} {LSTM} {FILENAME} {EXAMPLE_CORPUS_PATH} {MIN_LOSS} 1 {EPOCHS} 50000 {VERBOSE}'.split()
    args = parser.parse_args(command)
    with caplog.at_level(logging.INFO):
        args.callback(args)
    assert 'Loss after epoch' in caplog.text

    #generate text
    parser = ArgumentParser()
    setup_parser(parser)
    command = f'{GENERATE} {MODEL} {LSTM} {FILENAME} {DEFAULT_SAVEPATH} {VERBOSE}'.split()
    args = parser.parse_args(command)
    args.callback(args)
    import pdb; pdb.set_trace()
    assert len(capsys.readouterr().out) > 0
