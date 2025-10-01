"""
Configuration file for Enhanced A2C Maze Solver
Contains optimized hyperparameters and settings
"""

# Network Architecture
NETWORK_CONFIG = {
    'hidden_size': 256,  # Increased from 128
    'use_lstm': False,   # Can be enabled for memory-based tasks
    'use_attention': True,  # Self-attention mechanism
    'dropout_rate': 0.1,
    'num_heads': 4,  # For multi-head attention
}

# Training Hyperparameters
TRAINING_CONFIG = {
    'learning_rate': 1e-4,  # Lower for stability
    'gamma': 0.99,  # Discount factor
    'value_coef': 0.5,  # Value loss coefficient
    'entropy_coef': 0.02,  # Higher for more exploration
    'max_grad_norm': 0.5,  # Gradient clipping
    'weight_decay': 1e-5,  # L2 regularization
}

# Experience Replay
REPLAY_CONFIG = {
    'use_experience_replay': True,
    'replay_buffer_size': 15000,  # Increased buffer size
    'batch_size': 64,  # Larger batch for stability
    'replay_frequency': 1,  # Update every episode
}

# Exploration
EXPLORATION_CONFIG = {
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.995,
    'exploration_bonus': 0.2,  # Reward for new positions
    'revisit_penalty': -0.05,  # Penalty for revisiting
}

# Environment Settings
ENV_CONFIG = {
    'vision_range': 3,  # Local view radius
    'use_local_view': True,
    'max_steps': 300,  # Increased for complex mazes
    'reward_shaping': True,
}

# Curriculum Learning
CURRICULUM_CONFIG = {
    'episodes_per_level': [1500, 2000, 2500, 3000],  # Episodes for each level
    'success_threshold': 0.8,  # Success rate to advance
    'min_episodes_before_advance': 500,
    'render_frequency': 200,
}

# Reward Function Weights
REWARD_CONFIG = {
    'goal_reward': 100.0,
    'efficiency_bonus_weight': 50.0,
    'distance_reward_weight': 2.0,
    'movement_reward': 0.05,
    'wall_penalty': -0.2,
    'step_penalty': -0.02,
    'backtrack_penalty': -0.3,
    'time_pressure_weight': -0.01,
}

# Model Saving
MODEL_CONFIG = {
    'save_frequency': 1000,  # Save every N episodes
    'model_dir': 'models',
    'checkpoint_dir': 'checkpoints',
    'best_model_threshold': 0.9,  # Save as best if success rate > threshold
}

# Optimization Schedule
SCHEDULE_CONFIG = {
    'lr_schedule': True,
    'lr_step_size': 1000,
    'lr_gamma': 0.95,
    'target_update_frequency': 100,
}

# Logging and Monitoring
LOGGING_CONFIG = {
    'log_frequency': 50,
    'plot_frequency': 1000,
    'metrics_window': 100,  # Moving average window
    'verbose': True,
}

def get_optimized_config():
    """Get the complete optimized configuration"""
    return {
        'network': NETWORK_CONFIG,
        'training': TRAINING_CONFIG,
        'replay': REPLAY_CONFIG,
        'exploration': EXPLORATION_CONFIG,
        'environment': ENV_CONFIG,
        'curriculum': CURRICULUM_CONFIG,
        'reward': REWARD_CONFIG,
        'model': MODEL_CONFIG,
        'schedule': SCHEDULE_CONFIG,
        'logging': LOGGING_CONFIG,
    }

def print_config():
    """Print current configuration"""
    config = get_optimized_config()
    
    print("=== Enhanced A2C Configuration ===")
    for section, params in config.items():
        print(f"\n{section.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
