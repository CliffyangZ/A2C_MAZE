import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import intel_extension_for_pytorch as ipex
import os
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class Critic(nn.Module):
    def __init__(self, num_input: int, num_action, hidden_dim, checkpoint_dir='checkpoints', name='critic_network'):
        super(Critic, self).__init__()
        # Critic network 1
        self.fc1 = nn.Linear(num_input + num_action, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Critic network 2
        self.fc4 = nn.Linear(num_input + num_action, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, f'{name}.pth')

        self.apply(self._weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x))
        x1 = self.fc3(x)

        x2 = F.relu(self.fc4(x))
        x2 = F.relu(self.fc5(x))
        x2 = self.fc6(x)

        return x1, x2

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, num_input: int, num_action, hidden_dim, checkpoint_dir='checkpoints', name='actor_network', action_space=None):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_input, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, num_action)
        self.log_std_linear = nn.Linear(hidden_dim, num_action)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.apply(self._weights_init)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

class PredictiveModel(torch.nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, checkpoint_dir='checkpoints', name='predictive_network'):
        super(PredictiveModel, self).__init__()
        self.fc1 = torch.nn.Linear(num_inputs + num_actions, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_inputs)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        predicted_state = self.fc3(x)
        return predicted_state
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorCritic(nn.Module):
    """Enhanced Actor-Critic network for A2C algorithm with improved architecture"""
    
    def __init__(self, input_size: int, hidden_size: int, action_size: int, use_lstm: bool = False):
        super(ActorCritic, self).__init__()
        
        self.use_lstm = use_lstm
        self.hidden_size = hidden_size
        
        # Enhanced shared layers with residual connections
        self.shared_fc1 = nn.Linear(input_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        self.shared_fc3 = nn.Linear(hidden_size, hidden_size)
        self.shared_fc4 = nn.Linear(hidden_size, hidden_size)
        
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # LSTM for memory (optional)
        if self.use_lstm:
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.hidden_state = None
        
        # Enhanced Actor head with attention mechanism
        self.actor_fc1 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.actor_head = nn.Linear(hidden_size // 2, action_size)
        
        # Enhanced Critic head
        self.critic_fc1 = nn.Linear(hidden_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_head = nn.Linear(hidden_size // 2, 1)
        
        # Attention mechanism for better feature selection
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with improved initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use He initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def reset_hidden_state(self, batch_size: int = 1):
        """Reset LSTM hidden state"""
        if self.use_lstm:
            device = next(self.parameters()).device
            self.hidden_state = (
                torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device)
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the enhanced network"""
        batch_size = x.size(0)
        
        # Enhanced shared layers with residual connections
        x1 = F.relu(self.shared_fc1(x))
        if batch_size > 1:  # Only apply batch norm if batch size > 1
            x1 = self.bn1(x1)
        x1 = self.dropout(x1)
        
        x2 = F.relu(self.shared_fc2(x1))
        if batch_size > 1:
            x2 = self.bn2(x2)
        x2 = self.dropout(x2)
        
        # Residual connection
        x3 = F.relu(self.shared_fc3(x2)) + x1
        if batch_size > 1:
            x3 = self.bn3(x3)
        
        x4 = F.relu(self.shared_fc4(x3)) + x2  # Another residual connection
        
        # LSTM for temporal memory (if enabled)
        if self.use_lstm:
            if self.hidden_state is None:
                self.reset_hidden_state(batch_size)
            
            # Reshape for LSTM (batch_size, seq_len=1, features)
            lstm_input = x4.unsqueeze(1)
            lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)
            x4 = lstm_out.squeeze(1)
        
        # Self-attention mechanism
        # Reshape for attention (batch_size, seq_len=1, features)
        attn_input = x4.unsqueeze(1)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        x4 = attn_output.squeeze(1) + x4  # Residual connection
        
        # Enhanced Actor branch
        actor_x = F.relu(self.actor_fc1(x4))
        actor_x = F.relu(self.actor_fc2(actor_x))
        action_logits = self.actor_head(actor_x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Enhanced Critic branch
        critic_x = F.relu(self.critic_fc1(x4))
        critic_x = F.relu(self.critic_fc2(critic_x))
        state_value = self.critic_head(critic_x)
        
        return action_probs, state_value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Get action from the policy with optional deterministic mode"""
        action_probs, state_value = self.forward(state)
        
        if deterministic:
            # For testing, use the most probable action
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-8)
        else:
            # Sample action from probability distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, state_value