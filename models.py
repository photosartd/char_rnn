"""
File with RNN models for training
"""
import logging

import torch
import torch.nn as nn

LOGGER_NAME = 'Models logger'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(LOGGER_NAME)


class LSTModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 embedding_size,
                 device='cpu',
                 layers=1,
                 dropout_rate=0.2
                 ):
        """
        Init function for model
        :param input_size: number of tokens to choose from
        :param hidden_size: hidden size of LSTM
        :param embedding_size: each letters representation size
        :param device: 'cpu' or 'cuda'
        :param layers: number of layers in LSTM
        :param dropout_rate: dropout percentage
        """
        super(LSTModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.device = device
        self.layers = layers
        self.dropout_rate = dropout_rate

        self._embedding = nn.Embedding(self.input_size, self.embedding_size)
        self._lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.layers)
        self._dropout = nn.Dropout(self.dropout_rate)
        self._mlp = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        """
        Forward pass
        :param x: tensor [BATCH x SEQ_LEN x 1]
        :param hidden: tuple(tensor [LAYERS x BATCH x HIDDEN])
        :return: x, (h, c)
        """
        logger.debug('----------FORWARD PASS START----------')
        logger.debug(f'x shape: {x.shape}')
        logger.debug(f'hidden shape: {hidden[0].shape}')

        x = self._embedding(x).squeeze(2)

        logger.debug(f'x after embedding: {x.shape}')

        out, (h, c) = self._lstm(x, hidden)

        logger.debug(f'LSTM out shape: {out.shape}')

        out = self._dropout(out)
        x = self._mlp(out)

        logger.debug(f'Final out shape: {x.shape}')
        logger.debug('----------FORWARD PASS FINISH----------')

        return x, (h, c)

    def init_hidden(self, batch_size=1):
        """
        Initializes hidden with zeroes
        :param batch_size: batch size of hidden
        :return:
        """
        return (torch.zeros(self.layers, batch_size, self.hidden_size, requires_grad=True).to(self.device),
                torch.zeros(self.layers, batch_size, self.hidden_size, requires_grad=True).to(self.device))


class GRUModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 embedding_size,
                 device='cpu',
                 layers=1,
                 dropout_rate=0.2
                 ):
        """
        Init function for model
        :param input_size: number of tokens to choose from
        :param hidden_size: hidden size of LSTM
        :param embedding_size: each letters representation size
        :param device: 'cpu' or 'cuda'
        :param layers: number of layers in LSTM
        :param dropout_rate: dropout percentage
        """
        super(GRUModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.device = device
        self.layers = layers
        self.dropout_rate = dropout_rate

        self._embedding = nn.Embedding(self.input_size, self.embedding_size)
        self._gru = nn.GRU(self.embedding_size, self.hidden_size, self.layers)
        self._dropout = nn.Dropout(self.dropout_rate)
        self._mlp = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        """
        Forward pass
        :param x: tensor [BATCH x SEQ_LEN x 1]
        :param hidden: tensor [LAYERS x BATCH x HIDDEN]
        :return: x, h
        """
        logger.debug('----------FORWARD PASS START----------')
        logger.debug(f'x shape: {x.shape}')
        logger.debug(f'hidden shape: {hidden.shape}')

        x = self._embedding(x).squeeze(2)

        logger.debug(f'x after embedding: {x.shape}')

        out, h = self._gru(x, hidden)

        logger.debug(f'LSTM out shape: {out.shape}')

        out = self._dropout(out)
        x = self._mlp(out)

        logger.debug(f'Final out shape: {x.shape}')
        logger.debug('----------FORWARD PASS FINISH----------')

        return x, h

    def init_hidden(self, batch_size=1):
        """
        Initializes hidden with zeroes
        :param batch_size: batch size of hidden
        :return:
        """
        return torch.zeros(self.layers, batch_size, self.hidden_size, requires_grad=True).to(self.device)
