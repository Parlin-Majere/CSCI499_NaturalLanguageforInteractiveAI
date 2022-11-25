# IMPLEMENT YOUR MODEL CLASS HERE

import torch.nn as nn
import torch
import random


class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, input_dim, hidden_dim, embedding_dim, device):
        # will be using concanated inputs for the model
        super(Encoder, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim,embedding_dim)
        # some RNN, using LSTM here, bidirectional (maybe not needed)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim*2, hidden_dim)

        # Try to counteract overfitting
        self.dropout = nn.Dropout(0.4)


    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        #print("encode embeding",embedded.shape)
        #output not needed
        output, (hn, cn) = self.lstm(embedded)
        hn = self.fc_hidden(torch.concat((hn[0:1],hn[1:2]),dim=2))
        #print("hn shape: ",hn.shape)

        cn = self.fc_cell(torch.concat((cn[0:1],cn[1:2]),dim=2))

        return  output, hn, cn


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, output_dim, hidden_dim, embedding_dim, target_size, device):
        super(Decoder, self).__init__()
        self.device = device
        #self.attention = attention
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim 
        self.target_size = target_size
        # separate embeddings for action and target
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        #self.action_embedding = nn.Embedding(output_dim, embedding_dim)

        self.energy = nn.Linear(hidden_dim*3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(hidden_dim*2 + embedding_dim*2, hidden_dim, batch_first=False)

        # two separate prediciton heads
        self.tfc = nn.Linear(hidden_dim, target_size[1])
        self.afc = nn.Linear(hidden_dim, target_size[0])

        # try to counteract dropout
        self.dropout = nn.Dropout(0.4)


    def forward(self, tinput, ainput, hn, cn, encoder_outputs):
        #print("decoder tinput: ",tinput.shape)
        #print("decoder ainput: ",ainput.shape)
        if (tinput.type() != 'int'):
            tinput = tinput.unsqueeze(0)
            ainput = ainput.unsqueeze(0)
        tembedded = self.embedding(tinput)
        aembedded = self.embedding(ainput)

        tembedded = self.dropout(tembedded)
        aembedded = self.dropout(aembedded)

        embedded = torch.concat((aembedded,tembedded),dim=2)

        encoder_outputs = torch.transpose(encoder_outputs,0,1)
        sequence_length = encoder_outputs.shape[0]
        hn_reshape = hn.repeat(sequence_length,1,1)

        energy = self.relu(self.energy(torch.concat((hn_reshape, encoder_outputs), dim=2)))

        attention = self.softmax(energy)
        #print("attention shape: ", attention.shape)

        attention = attention.permute(1,2,0)
        #print("attention reshape: ",attention.shape)
        encoder_outputs = encoder_outputs.permute(1,0,2)
        #print("encoder reshape: ",encoder_outputs.shape)

        context = torch.bmm(attention,encoder_outputs).permute(1,0,2)

        lstm_input = torch.concat((context,embedded),dim=2)

        #a = self.attention(hn,encoder_outputs)

        #a = a.unsqueeze(1)

        #encoder_outputs = encoder_outputs.permute(1,0,1)

        #weighted = torch.bmm(a,encoder_outputs)

        #lstm_input = torch.concat((embedded,weighted),dim=2)

        #print("target embedding: ", embedded.shape)
        output,(hn,cn) = self.lstm(lstm_input, (hn,cn))
        #output,(hn,cn) = self.lstm(lstm_input,hn.unsqueeze(0))
        #print("decoder output: ",output.shape)
        #tprediction = self.tfc(output.squeeze(0))
        #aprediction = self.afc(output.squeeze(0))
        tprediction = self.tfc(output.squeeze(0))
        aprediction = self.afc(output.squeeze(0))
        #print("prediction shape: ",tprediction.shape)
        #print("predictions ",aprediction,tprediction)
        return tprediction, aprediction, hn, cn

class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, encoder, decoder, device):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_dim == decoder.hidden_dim

    def forward(self, input, target):
        #print("target shape: ",target.shape)
        #print("target[0] shape: ",target.shape)
        #print("encoderdecoder input: ", input.shape)
        #print(self.training)
        atarget = []
        ttarget = []
        for pairs in target:
            atarget.append(pairs[0].tolist())
            ttarget.append(pairs[1].tolist())
        
        batch_size = target.shape[0]
        target_length = target.shape[2]
        #print("ttarget shape: ",self.decoder.target_size)
        atarget_size = self.decoder.target_size[0]
        ttarget_size = self.decoder.target_size[1]
        #print("atarget/ttarget size: ", atarget_size, ttarget_size)
        toutputs = torch.zeros(target_length, batch_size, ttarget_size,device=self.device)
        aoutputs = torch.zeros(target_length, batch_size, atarget_size,device=self.device)
        #toutputs = torch.zeros(target_length, batch_size)
        #aoutputs = torch.zeros(target_length, batch_size)
        encoder_output, hn,cn = self.encoder(input)
        atarget = (torch.tensor(atarget,dtype=torch.long)).to(self.device)
        ttarget = (torch.tensor(ttarget,dtype=torch.long)).to(self.device)
        #print ("toutputs shape: ", toutputs.shape)
        #print(atarget,ttarget)
        #print("ainput shape ",atarget.shape[1])
        #print("tinput shape ",ttarget.shape[1])

        # take only the tokens
        ainput = atarget[:,0]
        tinput = ttarget[:,0]
        #print("input shapes", ainput.shape,tinput.shape)
        #print("ainput, tinput: ",ainput.shape,tinput.shape)
        #print(hn.shape)
        #print(cn.shape)

        #print("encoder ",encoder_output.shape,hn.shape,cn.shape,target_length)
        # default for global attention
        decoder_encoder_input = encoder_output

        # teacher enforcing / highest possibility
        for t in range(1,target_length):
            #print("tinput: ",tinput.shape)
            #print("ainput: ",ainput.shape)
            #print("hn: ", hn.shape)
            #print("cn: ", cn.shape)

            # defaulted to be global attention, here are the codes for local attention
            # partially mapping 380 -> 30
            start_index = (t-1)*12
            end_index = t*12
            decoder_encoder_input = encoder_output[:,start_index:end_index,:]

            toutput, aoutput, hn, cn = self.decoder(tinput,ainput,hn,cn,decoder_encoder_input)
            #print("shape of outputs: ", toutput.shape,aoutput.shape)
            #print(toutput, aoutput)
            # most likely word out of the dictionary
            toutputs[t] = toutput
            aoutputs[t] = aoutput
            #toutputs[t]=torch.argmax(toutput,dim=1)
            #aoutputs[t]=torch.argmax(aoutput,dim=1)
            #print("likelihood: ",aoutputs[t].shape,toutputs[t].shape)
            #top1 = output.argmax(1)
            # highest possibility
            #input = top1
            # if training == True, then teacher enforcing
            # adding on another condition to randomize teacher enforcing
            teacher_force = random.random() < 0.5 #teacher forcing ratio
            if self.training and teacher_force:
                ainput = (torch.tensor(atarget[:,t],dtype=torch.long)).to(self.device)
                #print("ainputs: ",ainput.shape,ainput)
                tinput = (torch.tensor(ttarget[:,t],dtype=torch.long)).to(self.device)
            else:
                #print("aoutputs: ",aoutputs.shape,aoutputs)
                aprediction = torch.argmax(aoutputs,-1)
                tprediction = torch.argmax(toutputs,-1)
                ainput = aprediction[t-1]
                tinput = tprediction[t-1]
                #print("ainput: ",ainput.shape,ainput)
            #print("new inputs: ",ainput.shape, tinput.shape)

        # transpose outputs
        #toutputs = toutputs.type(torch.IntTensor)
        #aoutputs = aoutputs.type(torch.IntTensor)
        aoutputs = torch.transpose(aoutputs,0,1)
        toutputs = torch.transpose(toutputs,0,1)
        #print("aoutput ",aoutputs.shape,aoutputs)
        aoutputs = torch.transpose(aoutputs,1,2)
        toutputs = torch.transpose(toutputs,1,2)
        #print ("final outputs: ", aoutputs.shape, toutputs.shape)
        #print(aoutputs,toutputs)
        return aoutputs,toutputs