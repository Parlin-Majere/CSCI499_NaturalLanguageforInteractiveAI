import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocabs_size):
        super(CBOW, self).__init__()
        self.vocab_size = vocabs_size
        self.embedding_dim = 128
        self.embed = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.embedding_dim,
        )
        self.linear1 = nn.Linear(self.embedding_dim,128)
        self.activation_function1 = nn.ReLU()
        self.linear2 = nn.Linear(128,self.vocab_size)
    def forward(self, input):
        embeds = self.embed(input)
        embeds = embeds.mean(axis=1)
        out1 = self.linear1(embeds)
        out1 = self.activation_function1(out1)
        out2 = self.linear2(out1)

        return out2