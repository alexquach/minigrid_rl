import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
import math

from mamba_ssm import Mamba, Block
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # Add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class TransformerCell(nn.Module):
    def __init__(self, d_model, nhead=16, dropout=0.1, num_layers=1, seq_len=16):
        super().__init__()
        self.seq_len = seq_len
        self.positional_encoder = PositionalEncoder(d_model, max_seq_len=self.seq_len)
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        # src: [B, L, D]
        src = self.positional_encoder(src)
        memory = torch.zeros_like(src)
        # if src_mask is None:
        #     src_mask = torch.zeros((self.seq_len, src.size(1)), device=src.device, dtype=torch.bool)
        output = self.transformer_decoder(src, memory)
        return output


class MambaCell(nn.Module):
    def __init__(self, d_model, seq_len=16):
        super(MambaCell, self).__init__()
        self.seq_len = seq_len
        self.mamba_block = Block(d_model, Mamba)
        self.d_model = d_model
        self.positional_encoder = PositionalEncoder(self.d_model)
        # self._init_weights()
    
    # def _init_weights(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def forward(self, src):
        # src: [B, L, D]
        src = self.positional_encoder(src)
        output, residual = self.mamba_block(src)  # Use input directly
        return output

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            if self.use_memory == "lstm":
                self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
            elif self.use_memory == "mamba":
                self.memory_rnn = MambaCell(self.image_embedding_size)
            elif self.use_memory == "transformer":
                self.memory_rnn = TransformerCell(self.image_embedding_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.embedding_size * self.memory_rnn.seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
            # nn.Linear(self.embedding_size, 64),
            # nn.Tanh(),
            # nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.embedding_size * self.memory_rnn.seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs_seq, mask_buffer=None):
        # [B, L, H, W, C]
        x = obs_seq.transpose(2, 4).transpose(3, 4)
        x = x.reshape(-1, *x.shape[2:])
        x = self.image_conv(x)
        x = x.reshape(obs_seq.shape[0], obs_seq.shape[1], -1)
        # [B, L, D]

        if self.use_memory == "transformer":
            embedding = self.memory_rnn(x, mask_buffer)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
