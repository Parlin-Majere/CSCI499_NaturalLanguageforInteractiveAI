# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import torch


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, input_dim, hidden_dim, embedding_dim):
        # will be using concanated inputs for the model
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim,embedding_dim)
        # some RNN, using LSTM here
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)


    def forward(self, x):
        embedded = self.embedding(x).view(1,1,-1)
        #output not needed
        output, (hn, cn) = self.lstm(embedded)

        return  hn, cn


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, output_dim, hidden_dim, embedding_dim, target_size):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.target_size = target_size
        # separate embeddings for action and target
        self.target_embedding = nn.Embedding(output_dim, target_size[0])
        self.action_embedding = nn.Embedding(output_dim, target_size[1])

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim,output_dim)


    def forward(self, input, hn, cn):
        input = input.unsqueeze(0)
        t_embedded = self.target_embedding(input)
        a_embedded = self.action_embedding(input)
        embedded = torch.concat(t_embedded,a_embedded)
        output,(hn,cn) = self.lstm(embedded, (hn,cn))
        prediction = self.fc(output.squeeze(0))
        return prediction, hn, cn

class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hidden_dim == decoder.hidden_dim

    def forward(self, input, target):
        batch_size = target.shape[1]
        target_length = target.shape[0]
        target_size = self.decoder.output_dim
        outputs = torch.zeros(target_length, batch_size, target_size)
        hn,cn = self.encoder(input)

        input = target[0,:]

        # teacher enforcing / highest possibility
        for t in range(1,target_length):
            output, hn, cn = self.decoder(input,hn,cn)
            outputs[t]=output
            top1 = output.argmax(1)
            # highest possibility
            #input = top1
            # teacher enforcing
            input = target[t]
        
        return outputs