import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Deque
from collections import deque
import random
from model import ActorCritic

class A2CAgent:
    """Enhanced Advantage Actor-Critic Agent with experience replay and improved features"""
    
    def __init__(self, input_size: int, hidden_size: int, action_size: int, 
                 learning_rate: float = 1e-4, gamma: float = 0.99, 
                 value_coef: float = 0.5, entropy_coef: float = 0.02,
                 use_lstm: bool = False, use_experience_replay: bool = True,
                 replay_buffer_size: int = 10000, batch_size: int = 32):
        
        self.device = torch.device("xpu" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.use_lstm = use_lstm
        self.use_experience_replay = use_experience_replay
        self.batch_size = batch_size
        
        # Network
        self.network = ActorCritic(input_size, hidden_size, action_size, use_lstm).to(self.device)
        self.target_network = ActorCritic(input_size, hidden_size, action_size, use_lstm).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Optimizers with different learning rates
        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Experience replay buffer
        if self.use_experience_replay:
            self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Training data storage
        self.reset_episode_data()
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        
        # Exploration parameters
        self.epsilon = 1.0  # For epsilon-greedy exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Target network update frequency
        self.target_update_freq = 100
        self.update_count = 0
        
    def reset_episode_data(self):
        """Reset episode data storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def get_action(self, state: np.ndarray) -> int:
        """Get action from the agent"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor)
        
        # Store for training (only store state here, action/reward stored separately)
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        return action
    
    def store_transition(self, action: int, reward: float, done: bool):
        """Store transition data"""
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_returns(self, next_value: float = 0.0) -> List[float]:
        """Compute discounted returns"""
        returns = []
        R = next_value
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def compute_advantages(self, returns: List[float]) -> List[float]:
        """Compute advantages using TD error"""
        advantages = []
        
        for i, (ret, value) in enumerate(zip(returns, self.values)):
            advantage = ret - value.item()
            advantages.append(advantage)
        
        return advantages
    
    def update(self, next_state: np.ndarray = None) -> float:
        """Update the network using collected episode data"""
        # Check if we have valid data to update
        if (len(self.rewards) == 0 or len(self.log_probs) == 0 or 
            len(self.states) == 0 or len(self.actions) == 0):
            self.reset_episode_data()
            return 0.0
        
        # Ensure all lists have the same length
        min_length = min(len(self.states), len(self.actions), len(self.rewards), 
                        len(self.log_probs), len(self.values), len(self.dones))
        
        if min_length == 0:
            self.reset_episode_data()
            return 0.0
        
        # Truncate all lists to the same length
        self.states = self.states[:min_length]
        self.actions = self.actions[:min_length]
        self.rewards = self.rewards[:min_length]
        self.log_probs = self.log_probs[:min_length]
        self.values = self.values[:min_length]
        self.dones = self.dones[:min_length]
        
        # Get next state value for bootstrapping
        next_value = 0.0
        if next_state is not None and not self.dones[-1]:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value_tensor = self.network(next_state_tensor)
                next_value = next_value_tensor.item()
        
        # Compute returns and advantages
        returns = self.compute_returns(next_value)
        advantages = self.compute_advantages(returns)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(self.states).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        log_probs_tensor = torch.stack(self.log_probs).to(self.device)
        values_tensor = torch.stack(self.values).squeeze().to(self.device)
        
        # Normalize advantages
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Compute losses
        # Actor loss (policy gradient)
        actor_loss = -(log_probs_tensor * advantages_tensor).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(values_tensor, returns_tensor)
        
        # Entropy loss for exploration
        action_probs, _ = self.network(states_tensor)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # Store loss for monitoring
        self.losses.append(total_loss.item())
        
        # Store experience in replay buffer
        if self.use_experience_replay:
            for i in range(len(self.states)):
                next_state = self.states[i+1] if i+1 < len(self.states) else None
                experience = (
                    self.states[i], self.actions[i], self.rewards[i],
                    next_state, self.dones[i], advantages[i], returns[i]
                )
                self.replay_buffer.append(experience)
        
        # Store individual losses for monitoring
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropy_losses.append(entropy_loss.item())
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        # Decay epsilon for exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Learning rate scheduling
        self.scheduler.step()
        
        # Reset episode data
        self.reset_episode_data()
        
        # Perform experience replay update if buffer is large enough
        if self.use_experience_replay and len(self.replay_buffer) >= self.batch_size:
            replay_loss = self.experience_replay_update()
            return total_loss.item() + replay_loss
        
        return total_loss.item()
    
    def experience_replay_update(self) -> float:
        """Perform experience replay update"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        next_states = [exp[3] for exp in batch]
        dones = torch.BoolTensor([exp[4] for exp in batch]).to(self.device)
        advantages = torch.FloatTensor([exp[5] for exp in batch]).to(self.device)
        returns = torch.FloatTensor([exp[6] for exp in batch]).to(self.device)
        
        # Get current action probabilities and values
        action_probs, values = self.network(states)
        
        # Calculate losses
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8).squeeze()
        
        # Actor loss
        actor_loss = -(action_log_probs * advantages).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy loss
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return total_loss.item()
    
    def get_action_with_exploration(self, state: np.ndarray, training: bool = True) -> int:
        """Get action with epsilon-greedy exploration"""
        if training and random.random() < self.epsilon:
            # Random action for exploration
            return random.randint(0, 3)  # Assuming 4 actions
        else:
            # Use policy
            return self.get_action(state)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        # PyTorch 2.6+: torch.load defaults to weights_only=True which can cause
        # UnpicklingError for checkpoints containing numpy objects. Since this
        # checkpoint was created by this project and includes optimizer state,
        # load with weights_only=False.
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.losses = checkpoint.get('losses', [])
        print(f"Model loaded from {filepath}")
    
    def plot_training_metrics(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            
            # Moving average
            if len(self.episode_rewards) > 10:
                window = min(100, len(self.episode_rewards) // 10)
                moving_avg = np.convolve(self.episode_rewards, 
                                       np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(self.episode_rewards)), 
                              moving_avg, 'r-', alpha=0.7, label=f'MA({window})')
                axes[0, 0].legend()
        
        # Episode lengths
        if self.episode_lengths:
            axes[0, 1].plot(self.episode_lengths)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
        
        # Losses
        if self.losses:
            axes[1, 0].plot(self.losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
        
        # Success rate (assuming reward > 50 means success)
        if self.episode_rewards:
            success_threshold = 50
            successes = [1 if r > success_threshold else 0 for r in self.episode_rewards]
            if len(successes) > 10:
                window = min(100, len(successes) // 10)
                success_rate = np.convolve(successes, np.ones(window)/window, mode='valid')
                axes[1, 1].plot(range(window-1, len(successes)), success_rate)
                axes[1, 1].set_title('Success Rate')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Success Rate')
                axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()