Char RNN for creation of texts
# Usage:
1) Clone this repository and make sure libraries from requirements.txt are installed
2) Prepare your text in .txt file
3) Open command line in repo
4) Command **./charrn.py train --filename /path/to/file.txt --verbose** makes RNN to train (may take long cos it trains on cpu by default)
5) Command **./charnn.py generate --filename model.pt --start your_start_words_with_underscore --verbose** makes the model to generate some text (200 symbols by default)
# Train arguments:
--filename - path to text for training; required<br>
--savepath - .pt filepath, where the model should be saved; default model.pt<br>
--verbose - training with verbosity; default false<br>
--model - **lstm** or **gru** options are available; default lstm<br>
--batch_size - batch size for training; default 16<br>
--seq_len - sequence len to train on; default 256<br>
--hidden_size - hidden layers; default 128<br>
--embedding_size - embedding size for symbols; default 128<br>
--device - device to train on; **cpu** or **cuda**; default cpu<br>
--layers - num of layers in RNN; default 2<br>
--epochs - num of epochs; default 2000<br>
--min_loss - min loss to train until; default 0.56 (to not to overfit)<br>

# Generate arguments:
--filename - /path/to/model.pt file<br>
--model - **lstm** or **gru**; pass same as in training; default **lstm**<br>
--verbose<br>
--len - text len to generate; default 200<br>
--start - start text to generate (with_underscores to parse)<br>
--temp - temperature of text generation; the more, the more different from the original results should be observed; default 1.0 (no randomness)<br>
