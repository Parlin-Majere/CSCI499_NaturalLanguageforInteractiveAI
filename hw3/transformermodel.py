import torch
import torch.nn as nn

# Model altered from Aladdin Persson's implementation of Encoder Decoder model
# with transformers, wanted to use huggingface but alternating them to be 
# outputing two sequences prove to be a pain for one person in desperation to finish
# the project and have never used huggingface before. I've considered running two
# models at the same time with different encoder for outputing target and action,
# but I think that kind of defeat the purpose of this assignment.

class SelfAttention (nn.Module):
    def __init__(self, embedding_dim, heads):
        super(SelfAttention,self).__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim//heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim,embedding_dim)
    
    def forward(self,values,keys,query,mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = values.reshape(N, key_len, self.heads, self.head_dim)
        queries = values.reshape(N, key_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])

        if mask is not None:
            energy = energy.masked_fill(mask==0,float("-1e20"))

        attention = torch.softmax(energy/(self.embedding_dim ** (1/2)),dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention,values]).reshape(N,query_len,self.heads*self.head_dim)

        out = self.fc_out(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion):
        super(TransformerBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self,value,key,query,mask):
        attention = self.attention(value,key,query,mask)

        x = self.dropout(self.norm1(attention+query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))

        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder,self).__init__()
        self.embed_size=embed_size
        self.device=device
        self.word_embedding = nn.Embedding(src_vocab_size,embed_size).to(device)
        self.position_embedding = nn.Embedding(max_length,embed_size).to(device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion = forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,mask):
        N, seq_length = x.shape
        positions = torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out,out,out,mask)
        
        return out

class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,forward_expansion,dropout,device):
        super(DecoderBlock,self).__init__()
        self.norm=nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size,heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(value,key,query,src_mask)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
    ):
        super(Decoder,self).__init__()
        self.device = device
        self.word_embed_size = embed_size//2
        self.word_embedding = nn.Embedding(trg_vocab_size[0]+trg_vocab_size[1],embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size,heads,forward_expansion,dropout,device).to(device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out1 = nn.Linear(embed_size,trg_vocab_size[0])
        self.fc_out2 = nn.Linear(embed_size,trg_vocab_size[1])
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,enc_out,src_mask,trg_mask):
        # what differs is the need to split up x, which is input as we are inputting two sequences
        atarget = []
        ttarget = []
        for pairs in x:
            atarget.append(pairs[0].tolist())
            ttarget.append(pairs[1].tolist())
        atarget = (torch.tensor(atarget,dtype=torch.long)).to(self.device)
        ttarget = (torch.tensor(ttarget,dtype=torch.long)).to(self.device)

        # all these kind of parameters should be the same but just doing it to save thinking
        aN, aseq_length = atarget.shape
        tN, tseq_length = ttarget.shape
        apositions = torch.arange(0,aseq_length).expand(aN,aseq_length).to(self.device)
        tpositions = torch.arange(0,tseq_length).expand(tN,tseq_length).to(self.device)
        aembedding = self.word_embedding(atarget)
        tembedding = self.word_embedding(ttarget)
        embedding = torch.concat((aembedding,tembedding),dim=0)
        #print(embedding.shape,self.position_embedding(apositions).shape)
        ax = self.dropout(aembedding+self.position_embedding(apositions))
        tx = self.dropout(tembedding+self.position_embedding(tpositions))

        aout = self.fc_out1(ax)
        tout = self.fc_out2(tx)

        return aout, tout

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.2,
        device="cpu",
        max_length=400,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        atarget = []
        ttarget = []
        for pairs in trg:
            atarget.append(pairs[0].tolist())
            ttarget.append(pairs[1].tolist())
        atarget = (torch.tensor(atarget,dtype=torch.long)).to(self.device)
        ttarget = (torch.tensor(ttarget,dtype=torch.long)).to(self.device)
        #print(atarget.shape,ttarget.shape)
        N, trg_len = atarget.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        #print("atarget and ttarget masks: ", atrg_mask.shape,ttrg_mask.shape)

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        aout,tout = self.decoder(trg, enc_src, src_mask, trg_mask)
        return aout,tout