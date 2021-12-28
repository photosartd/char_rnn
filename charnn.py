#!/usr/bin/env python
"""
Program that allows to use RNN to teach it how to write texts similar to given corpus
"""
from argparse import ArgumentParser, Namespace
import logging

import pickle as pkl
import torch
import torch.nn as nn

from data_utilities import read_text, text_to_seq, train, evaluate
from models import LSTModel, LOGGER_NAME
from constants import *


logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)


def on_train(args: Namespace):
    """
    Model training
    :param args: Namespace with arguments
    :return:
    """
    if not args.verbose:
        logger.disabled = True
    sequence, token_to_idx, idx_to_token = text_to_seq(read_text(args.filename))
    filename = args.savepath.split('/')[-1].split('.')[0]
    pkl.dump(token_to_idx, open(f'{filename}_{TOKEN_TO_IDX}{PKL}', 'wb'))
    pkl.dump(idx_to_token, open(f'{filename}_{IDX_TO_TOKEN}{PKL}', 'wb'))
    input_size = len(token_to_idx)
    device = args.device
    model = None
    if args.model == LSTM:
        model = LSTModel(
            input_size=input_size,
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            device=device,
            layers=args.layers
        )
    elif args.model == GRU:
        raise NotImplementedError()
    else:
        raise Exception(f'Wrong {MODEL} argument passed')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=OPTIMIZER_LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        verbose=True,
        factor=0.5
    )
    epochs = args.epochs
    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        sequence=sequence,
        idx_to_token=idx_to_token,
        token_to_idx=token_to_idx,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        epochs=epochs,
        device=device,
        scheduler=scheduler,
        savepath=args.savepath,
        min_loss=args.min_loss
    )


def on_generate(args: Namespace):
    """
    Text generation
    :param args: Namespace
    :return:
    """
    try:
        name = args.filename.split('/')[-1].split('.')[0]
        tokens_filename = name + '_' + TOKEN_TO_IDX + PKL
        idx_filename = name + '_' + IDX_TO_TOKEN + PKL
        token_to_idx = pkl.load(open(tokens_filename, 'rb'))
        idx_to_token = pkl.load(open(idx_filename, 'rb'))
        input_size = len(token_to_idx)
        model = LSTModel(
            input_size,
            args.hidden_size,
            embedding_size=args.embedding_size,
            device=args.device,
            layers=args.layers,
        )
        model.load_state_dict(torch.load(args.filename))
        start = ' '.join(args.start.split('_'))
        res = evaluate(
            model,
            token_to_idx,
            idx_to_token,
            device=args.device,
            start_text=start,
            temperature=args.temp,
            length=args.len,
        )
        print(res)
    except Exception:
        logger.warning(LOAD_WARNING)


def setup_parser(parser: ArgumentParser):
    """
    Sets up our argument parser's params
    :param parser: ArgumentParser
    :return: None
    """
    parser.add_argument(ACTION, help=ACTION_HELP)
    parser.add_argument(MODEL, default=LSTM, choices=[LSTM, GRU], help=MODEL_HELP)
    parser.add_argument(FILENAME, help=FILENAME_HELP, required=True)
    parser.add_argument(SAVEPATH, default=DEFAULT_SAVEPATH, help=SAVEPATH_HELP)
    parser.add_argument(BATCH_SIZE, default=DEFAULT_BATCH_SIZE, const=DEFAULT_BATCH_SIZE, nargs='?', type=int)
    parser.add_argument(SEQ_LEN, default=DEFAULT_SEQ_LEN, const=DEFAULT_SEQ_LEN, nargs='?', type=int)
    parser.add_argument(HIDDEN_SIZE, default=DEFAULT_HIDDEN_SIZE, const=DEFAULT_HIDDEN_SIZE, nargs='?', type=int,
                        help=HIDDEN_SIZE_HELP)
    parser.add_argument(EMBEDDING_SIZE, default=DEFAULT_EMBEDDING_SIZE, const=DEFAULT_EMBEDDING_SIZE, nargs='?', type=int)
    parser.add_argument(DEVICE, default=DEFAULT_DEVICE, const=DEFAULT_DEVICE, nargs='?', type=str, help=DEVICE_HELP)
    parser.add_argument(LAYERS, default=DEFAULT_LAYERS, const=DEFAULT_LAYERS, nargs='?', type=int)
    parser.add_argument(EPOCHS, default=DEFAULT_EPOCHS, const=DEFAULT_EPOCHS, nargs='?', type=int, help=EPOCHS_HELP)
    parser.add_argument(LEN, default=DEFAULT_LEN, const=DEFAULT_LEN, nargs='?', type=int, help=LEN_HELP)
    parser.add_argument(START, default=' ', const=' ', nargs='?', type=str, help=START_HELP)
    parser.add_argument(TEMP, default=DEFAULT_TEMP, const=DEFAULT_TEMP, nargs='?', type=float, help=TEMP_HELP)
    parser.add_argument(MIN_LOSS, default=DEFAULT_MIN_LOSS, const=DEFAULT_MIN_LOSS, nargs='?', type=float,)
    parser.add_argument(VERBOSE, action='store_true', help=VERBOSE_HELP, default=False,)
    parser.set_defaults(callback=process_args)


def process_args(args: Namespace):
    """Executes all arguments given to the parser"""
    if args.action == TRAIN:
        on_train(args)
    elif args.action == GENERATE:
        on_generate(args)
    else:
        raise NotImplementedError()


def main():
    """
    Main program
    :return: None
    """
    parser = ArgumentParser(
        description=HELP_STRING
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
