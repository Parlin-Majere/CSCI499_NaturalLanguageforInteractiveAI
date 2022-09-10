# IMPLEMENT YOUR MODEL CLASS HERE
import torch.nn as nn
import torch.nn.functional as F

class AlfredLSTM(nn.Module):
    def __init__ (self, embedding_dim, hidden_dim, vocab_size, target_sizes):
        super(AlfredLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        #print(vocab_size)
        #print(target_sizes)
        self.word_action_embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.word_target_embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim*60, hidden_dim)
        self.hidden2atag=nn.Linear(hidden_dim,target_sizes[0])
        self.hidden2ttag=nn.Linear(hidden_dim,target_sizes[1])
        #print("model initialized")
    
    def forward(self,sentence,tags):

        #print("forward")
        #print(len(sentence))

        aembeds = self.word_action_embeddings(sentence)
        tembeds = self.word_target_embeddings(sentence)

        #print(aembeds.view(len(sentence),1,-1))

        alstm_out, _ = self.lstm(aembeds.view(len(sentence),1,-1))
        tlstm_out, _ = self.lstm(tembeds.view(len(sentence),1,-1))

        atag_space = self.hidden2atag(alstm_out.view(len(sentence),-1))
        ttag_space = self.hidden2ttag(tlstm_out.view(len(sentence),-1))

        atags_scores = F.log_softmax(atag_space,dim=1)
        ttags_scores = F.log_softmax(ttag_space,dim=1)
        
        return atags_scores,ttags_scores