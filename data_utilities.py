"""
Some utilities for data preparation
"""
import logging
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from models import LOGGER_NAME
from constants import *


logger = logging.getLogger(LOGGER_NAME)


def read_text(filename: str, sep: str = ''):
    """
    Reads file, lowercases it and joins using sep
    :param sep: separator to join strings in file
    :param filename: name of the file with corpus
    :return: str - text
    """
    with open(filename, 'r') as file:
        text = file.readlines()
    return sep.join(text).lower()


def text_to_seq(text: str):
    """
    converts text to sequence of encoded letters
    :param text: str
    :return: sequence, token_to_idx, idx_to_token
    """
    char_cnt = Counter(text)
    char_cnt = sorted(char_cnt.items(), key=lambda x: x[1], reverse=True)
    sorted_chars = [char for char, _ in char_cnt]
    token_to_idx = {char: idx for idx, char in enumerate(sorted_chars)}
    idx_to_token = {val: key for key, val in token_to_idx.items()}
    sequence = np.array([token_to_idx[token] for token in text])

    return sequence, token_to_idx, idx_to_token


def get_batch(sequence: np.ndarray, batch_size: int, seq_len: int):
    """
    Creates batch of size batch_size
    :param sequence: encoded text
    :param batch_size: size of the batch
    :param seq_len: length of one sentence
    :return: batch, labels
    """
    data = []
    labels = []
    for _ in range(batch_size):
        start = np.random.randint(0, len(sequence) - seq_len)
        chunk = sequence[start: start + seq_len]
        curr_data = torch.LongTensor(chunk[:-1]).view(-1, 1)
        curr_labels = torch.LongTensor(chunk[1:]).view(-1, 1)
        data.append(curr_data)
        labels.append(curr_labels)
    return torch.stack(data, dim=0), torch.stack(labels, dim=0)


def train(model,
          optimizer,
          criterion,
          sequence,
          idx_to_token,
          token_to_idx,
          batch_size=DEFAULT_BATCH_SIZE,
          seq_len=DEFAULT_SEQ_LEN,
          epochs=DEFAULT_EPOCHS,
          device=DEFAULT_DEVICE,
          scheduler=None,
          min_loss=DEFAULT_MIN_LOSS,
          savepath=None
          ):
    """
    Model training function
    :param model: nn.Module
    :param optimizer: optimizer from torch.optim
    :param criterion: e.g. CrossEntropyLoss
    :param sequence: encoded text
    :param idx_to_token: dict
    :param token_to_idx: dict
    :param batch_size: int
    :param seq_len: int
    :param epochs: int
    :param device: 'cpu' or 'cuda'
    :param scheduler: from torch.optim.lr_scheduler
    :param min_loss: loss to end training
    :param savepath: path to save model
    :return:
    """
    losses_history = []
    for epoch in range(epochs):
        logger.debug('**********EPOCH START**********')
        model.train()
        #batch_size x seq_len
        train, target = get_batch(sequence, batch_size, seq_len)

        logger.debug(f'train shape before permuting: {train.shape}')

        train = train.permute(1, 0, 2).to(device)
        target = target.permute(1, 0, 2).to(device)

        logger.debug(f'train shape after permuting: {train.shape}')

        hidden = model.init_hidden(batch_size)

        #main loop
        output, hidden = model(train, hidden)
        output = output.permute(1, 2, 0)
        target = target.squeeze(-1).permute(1, 0)

        logger.debug(f'output shape after permuting: {output.shape}')
        logger.debug(f'target shape after squeeze and permuting: {target.shape}')

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses_history.append(loss.item())

        if len(losses_history) >= 50:
            avg_loss = np.mean(losses_history)
            logger.info(f'Loss after epoch {epoch}: {avg_loss}')
            if scheduler:
                scheduler.step(avg_loss)
            losses_history = []
            model.eval()
            predicted_text = evaluate(model, token_to_idx, idx_to_token, device)
            logger.info(predicted_text)
            logger.info(f'Current loss: {avg_loss}')
            if avg_loss < min_loss:
                logger.info('Minimal loss reached; saving model...')
                torch.save(model.state_dict(), savepath)
                break

        logger.debug('**********EPOCH END**********')


def evaluate(
        model,
        token_to_idx: dict,
        idx_to_token: dict,
        device,
        start_text=' ',
        length=100,
        temperature=1.0
):
    """
    Evaluates trained model based on start_text
    :param model: trained model
    :param token_to_idx: dict
    :param idx_to_token: dict
    :param device: 'cpu' or 'cuda'
    :param start_text: text to start prediction
    :param length: length of prediction
    :param temperature: how random prediction is
    :return: predicted text
    """
    hidden = model.init_hidden()
    idx_input = [token_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    _, hidden = model(train, hidden)

    inp = train[-1].view(-1, 1, 1)

    for i in range(length):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temperature, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(token_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_token[top_index]
        predicted_text += predicted_char

    return predicted_text
