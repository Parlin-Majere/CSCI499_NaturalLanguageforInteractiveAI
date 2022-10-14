import torch.nn as nn

class CBOW(nn.Module):
    vocab_size = 0
    embedding_dim = 128
    def __init__(self, vocabs_size):
        super(CBOW, self).__init__()
        vocab_size = vocabs_size
        self.embeddings = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = 128,
            max_norm = 1
        )
        self.linear = nn.Linear(
            in_features = 128,
            out_features = vocab_size
        )
    def forward(self, input):
        x = self.embeddings(input)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x