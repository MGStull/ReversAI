import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed Size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask, illegal_mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        if illegal_mask is not None:
            if illegal_mask.dim() == 2:
                illegal_mask = illegal_mask.unsqueeze(1).unsqueeze(1)
            energy = energy.masked_fill(illegal_mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, illegal_mask=None):
        # Attention with residual connection and layer norm
        attention = self.attention(value, key, query, mask, illegal_mask)
        x = self.dropout(attention)
        x = self.norm1(x + query)  # Residual connection
        
        # Feed-forward with residual connection and layer norm
        forward = self.feed_forward(x)
        forward = self.dropout(forward)
        out = self.norm2(forward + x)  # Residual connection
        
        return out


class ReversiBotDecoder(nn.Module):
    def __init__(
        self,
        vocab_size=64,
        embed_size=512,
        num_layers=8,
        heads=16,
        dropout=0.1,
        device='cuda',
        max_length=60,
        forward_expansion=4  # REDUCED from 8
    ):
        super(ReversiBotDecoder, self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_length = max_length

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.turn_embedding = nn.Embedding(3, embed_size)
        
        # Scale embeddings to prevent early training instability
        self.embedding_scale = embed_size ** 0.5
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size,
                heads,
                dropout,
                forward_expansion
            )
            for _ in range(num_layers)
        ])

    def forward(self, move_sequence, turns=None, illegal_moves=None):
        N, seq_length = move_sequence.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        # Scale embeddings to prevent magnitude explosion
        embed_moves = self.word_embedding(move_sequence) * self.embedding_scale
        embedded_positions = self.position_embedding(positions)
        
        if turns is not None:
            turns_adjusted = turns.clone()
            turns_adjusted[turns == -1] = 1
            turns_adjusted[turns == 1] = 2
            turns_adjusted[turns == 0] = 0
            embed_turns = self.turn_embedding(turns_adjusted.long())
            x = self.dropout_layer(embed_moves + embedded_positions + embed_turns)
        else:
            x = self.dropout_layer(embed_moves + embedded_positions)
        
        casual_mask = self.make_casual_mask(seq_length, N)

        for layer in self.layers:
            x = layer(x, x, x, casual_mask, illegal_mask=None)
        
        logits = self.fc_out(x)

        if illegal_moves is not None:
            illegal_moves = illegal_moves.to(self.device)
            illegal_moves_expanded = illegal_moves.unsqueeze(1)
            logits = logits.masked_fill(illegal_moves_expanded == 0, float("-1e20"))
        
        return logits

    def make_casual_mask(self, seq_length, batch_size):
        mask = torch.tril(torch.ones(seq_length, seq_length))
        mask = mask.expand((batch_size, 1, seq_length, seq_length))
        mask = mask.to(self.device)
        return mask