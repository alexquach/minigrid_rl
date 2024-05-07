from einops import rearrange
import torch

import utils
from .other import device
from model import ACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir, filename="status.pt",
                 argmax=False, num_envs=1, use_memory=False, use_text=False, seq_len=16):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text, seq_len=seq_len)
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)

        self.acmodel.load_state_dict(utils.get_model_state(model_dir, filename))
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

        self.obs_buffer = torch.zeros(self.acmodel.memory_rnn.seq_len, self.num_envs, *obs_space["image"], device=device)

    def get_actions(self, obss):
        with torch.no_grad():
            preprocessed_obs_seq = rearrange(self.obs_buffer, 'l b h w c -> b l h w c')
            if self.acmodel.use_memory == "lstm":
                preprocessed_obs_seq = preprocessed_obs_seq.squeeze(1)
                dist, _, self.memories = self.acmodel(preprocessed_obs_seq, memory=self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obs_seq, None)

        preprocessed_obs = self.preprocess_obss(obss).image # [B, H, W, C] 

        # Update buffers [L, B, H, W, C]
        self.obs_buffer = torch.roll(self.obs_buffer, -1, dims=0)
        self.obs_buffer[-1] = preprocessed_obs

        preprocessed_obs_seq = rearrange(self.obs_buffer, 'l b h w c -> b l h w c')
        # Do one agent-environment interaction
        with torch.no_grad():
            if self.acmodel.use_memory == "lstm":
                preprocessed_obs_seq = preprocessed_obs_seq.squeeze(1)
                dist, _, self.memories = self.acmodel(preprocessed_obs_seq, memory=self.memories)
            else:
                dist, value, embedding = self.acmodel(preprocessed_obs_seq, None, return_hidden=True)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        if self.acmodel.use_memory == "lstm":
            return actions.cpu().numpy(), self.memories.cpu().numpy()
        return actions.cpu().numpy(), embedding.cpu().numpy()

    def get_action(self, obs):
        actions, embeddings = self.get_actions([obs])
        return actions[0], embeddings[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
