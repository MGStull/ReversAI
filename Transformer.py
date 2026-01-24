# This Code was followed based on a tutorial by Aladdin Persson https://www.youtube.com/watch?v=U0s0f995w14 on the paper attention is all you need
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size,heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads

        assert (self.head_dim * heads == embed_size), "Embed Size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias =  False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)

        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self,values,keys,query,mask, illegal_mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values  = values.reshape(N, value_len, self.heads, self.head_dim)
        keys    = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        # queries shape: (N,query_len, heads, heads_dim)
        # keys shape:    (N, key_len,head,heads_dim)
        # energy shape:  (N, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        if illegal_mask is not None:
            if illegal_mask.dim() == 2:
                illegal_mask = illegal_mask.unsqueeze(1).unsqueeze(1)
            energy = energy.masked_fill(illegal_mask == 0, float("-1e20"))

        attention = torch.softmax(energy/(self.embed_size**(1/2)),dim=3)

        out = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N,value_len, heads, heads_dim)
        # after einsum (N, query_len, heads, head_dim) then flattern last two dimensions
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self,embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query, mask, illegal_mask = None):
        attention = self.attention(value, key, query, mask, illegal_mask)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out

class ReversiBotDecoder(nn.Module):
    def __init__(self, vocab_size=64, embed_size=128, num_layers=4, heads=8,dropout=0.1,device='cuda',max_length=60, forward_expansion=4):
        super(ReversiBotDecoder,self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_length = max_length

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding =  nn.Embedding(max_length,embed_size)
        self.turn_embedding = nn.Embedding(3,embed_size)
        self.fc_out = nn.Linear(embed_size,vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size,
                heads,
                dropout,
                forward_expansion
            ) for _ in range(num_layers)
        ])

    def forward(self, move_sequence, turns=None,illegal_moves=None):
        N,seq_length = move_sequence.shape
        positions = torch.arange(0, seq_length).expand(N,seq_length).to(self.device)
        embed_moves = self.word_embedding(move_sequence)
        embedded_positions = self.position_embedding(positions)
        if turns is not None:
            turns_adjusted = turns.clone()
            turns_adjusted[turns == -1] = 1
            turns_adjusted[turns == 1] = 2
            turns_adjusted[turns == 0] = 0

            embed_turns = self.turn_embedding(turns_adjusted.long())
            x = self.dropout_layer(embed_moves + embedded_positions + embed_turns)
        else: 
            x = self.dropout_layer(embed_moves + embedded_positions + embed_turns)
        casual_mask = self.make_casual_mask(seq_length,N)

        for layer in self.layers:
            x = layer(x, x, x, casual_mask, illegal_mask=None)
        logits = self.fc_out(x)

        if illegal_moves is not None:
            illegal_moves_expanded = illegal_moves
            logits = logits.masked_fill(illegal_moves_expanded==0, float("-1e20"))
            
        return logits

    def make_casual_mask(self, seq_length, batch_size):
        mask = torch.tril(torch.ones(seq_length, seq_length))
        mask = mask.expand((batch_size,1,seq_length,seq_length))
        mask = mask.to(self.device)
        return mask


