import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac

from mamba_ssm import Mamba, Block
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class TransformerCell(nn.Module):
    def __init__(self, d_model, nhead=16, dim_feedforward=2048, dropout=0.1):
        super(TransformerCell, self).__init__()
        self.transformer_block = TransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                                         dim_feedforward=dim_feedforward, dropout=dropout)
        self.state_buffer = None

    def forward(self, tgt, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, transformer_state=None):
        if transformer_state is None:
            transformer_state = self.init_states(batch_size=tgt.size(0), d_model=tgt.size(1))

        transformer_state = transformer_state.unsqueeze(1)
        
        output = self.transformer_block(tgt, transformer_state, tgt_mask=tgt_mask, memory_mask=memory_mask, 
                                        tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        self.state_buffer = output  # Update state buffer with the latest output
        return output, self.state_buffer

        # def forward(self, src, src_mask=None, src_key_padding_mask=None, transformer_state=None):
        # if transformer_state is None:
            # transformer_state = self.init_states(batch_size=src.size(0), d_model=src.size(1))
        
        # output = self.transformer_block(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def init_states(self, batch_size, d_model):
        print(f"dmodel: {d_model}")
        return torch.zeros(batch_size, 1, d_model * 1)

class MambaCell(nn.Module):
    def __init__(self, d_model, ssm_state_size):
        super(MambaCell, self).__init__()
        self.ssm = Block(ssm_state_size, Mamba)
        self.d_model = d_model
        self.ssm_state_size = ssm_state_size

    def forward(self, input, ssm_state=None):
        if ssm_state is None:
            ssm_state = self.init_states(batch_size=input.size(0))

        # print(f"ssm_state: {ssm_state.shape}")
        ssm_output, new_ssm_state = self.ssm(input)  # Use input directly
        return ssm_output, new_ssm_state

    def init_states(self, batch_size):
        ssm_state = torch.zeros(batch_size, 1, self.ssm_state_size * 2)
        return ssm_state

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
                self.memory_rnn = MambaCell(self.image_embedding_size, self.semi_memory_size)
            elif self.use_memory == "transformer":
                self.memory_rnn = TransformerCell(self.image_embedding_size, self.semi_memory_size)

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
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
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

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory == "lstm":
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        elif self.use_memory == "mamba":
            memory = memory.view(x.shape[0], 1, -1)
            hidden = memory
            embedding, residual = self.memory_rnn(x, hidden)
            embedding = embedding.view(x.shape[0], -1)
            memory = torch.cat([embedding, residual], dim=1)
        elif self.use_memory == "transformer":
            x = x.unsqueeze(1)
            embedding, memory = self.memory_rnn(x, transformer_state=memory)
            embedding = embedding.squeeze(1)
            memory = memory.squeeze(1)
            memory = memory.view(x.shape[0], -1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
