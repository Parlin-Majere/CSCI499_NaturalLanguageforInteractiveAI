Something to note

1. Again, there is a problem with training the model on GPU. Apparently the initial setup does not allow every tensor
to be loaded onto GPU, some manipulation is needed at least for me to make the thing run on GPU. Maybe this will be something
to look for in the future when building the initial framework for the model.


Introduction to assignment structures

The hw3 folder here contains 2 training code (train.py, transformertrain.py), three models (model.py, attnmodel.py, transformermodel.py)
corresponding to Encoder-Decoder LSTM model, Encoder-Attention Decoder LSTM model, and a transformer model. Then, there are a few folders
containing the training tracking pdf's, which you can defer from name what model they correspond to.

For the encoder-decoder model, instead of making encoder and decoder just the LSTM/RNN models they are as some structure explained,
I still chose to put everything they need (hidden layer, embedding layer, fc layers and etc) in there, not because doing this has an
advantage over other methods, it is because it preserves the structure from hw1.

For data preprocessing, I choose to do it as 3 sequences. For each episode which contains number of instruction -> (action, target)
pairings, 