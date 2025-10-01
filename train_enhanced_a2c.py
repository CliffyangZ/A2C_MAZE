import numpy as np
import matplotlib.pyplot as plt
import time
import os
from gym_custom import MazeEnv
from agent import A2CAgent


def create_curriculum_mazes():
    """Create a curriculum of mazes from easy to hard"""
    
    # Very simple 5x5 maze
    maze_5x5 = [
        [1,1,1,1,1,1,1],
        [1,3,0,0,0,2,1],
        [1,0,1,1,1,0,1],
        [1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1]
    ]
    
    # Simple 8x8 maze
    maze_8x8 = [
        [1,1,1,1,1,1,1,1,1,1],
        [1,3,0,0,1,0,0,0,2,1],
        [1,0,1,0,1,0,1,1,0,1],
        [1,0,1,0,0,0,1,0,0,1],
        [1,0,0,0,1,0,0,0,1,1],
        [1,1,1,0,1,1,1,0,0,1],
        [1,0,0,0,0,0,0,0,1,1],
        [1,1,1,1,1,1,1,1,1,1]
    ]
    
    # Medium 12x12 maze
    maze_12x12 = [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,3,0,0,0,0,0,1,0,0,0,0,2,1],
        [1,1,1,1,0,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,0,1,0,0,0,1,0,0,0,1],
        [1,0,1,1,1,1,1,1,0,1,0,1,1,1],
        [1,0,0,0,0,0,0,0,0,1,0,0,0,1],
        [1,1,1,0,1,1,1,1,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,1,0,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ]
    
    # Complex maze (from original)
    maze_complex = [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,3,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1],
        [1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,1,1],
        [1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1],
        [1,1,0,0,0,0,0,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,0,0,1,1],
        [1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1],
        [1,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,0,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,0,0,1,1,1],
        [1,1,1,1,0,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,1,0,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,0,1,1,1,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ]
    
    return {
        'level1': maze_5x5,
        'level2': maze_8x8,
        'level3': maze_12x12,
        'level4': maze_complex
    }


def train_with_curriculum(curriculum_mazes, episodes_per_level=2000, render_every=200):
    """Train agent using curriculum learning"""
    
    print("=== Enhanced A2C Training with Curriculum Learning ===")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    agent = None
    all_rewards = []
    all_lengths = []
    
    for level, (maze_name, maze_layout) in enumerate(curriculum_mazes.items(), 1):
        print(f"\n🎯 Training Level {level}: {maze_name}")
        print(f"Maze size: {len(maze_layout)}x{len(maze_layout[0])}")
        
        # Create environment for this level
        env = MazeEnv(maze_layout, render_mode="human", vision_range=3, use_local_view=True)
        
        # Create or update agent
        if agent is None:
            # First level - create new agent
            input_size = env.observation_space.shape[0]
            hidden_size = 256
            action_size = env.action_space.n
            
            print(f"Creating agent: input={input_size}, hidden={hidden_size}, actions={action_size}")
            
            agent = A2CAgent(
                input_size=input_size,
                hidden_size=hidden_size,
                action_size=action_size,
                learning_rate=2e-4,  # Slightly higher for curriculum learning
                gamma=0.99,
                value_coef=0.5,
                entropy_coef=0.03,  # Higher entropy for exploration
                use_lstm=False,
                use_experience_replay=True,
                replay_buffer_size=15000,
                batch_size=64
            )
        else:
            # Reset LSTM hidden state if using LSTM
            if agent.use_lstm:
                agent.network.reset_hidden_state()
        
        # Training loop for this level
        level_rewards = []
        level_lengths = []
        success_count = 0
        
        for episode in range(episodes_per_level):
            state, info = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Get action with exploration
                action = agent.get_action_with_exploration(state, training=True)
                
                # Take step
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition
                agent.store_transition(action, reward, done)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                # Render occasionally
                if episode % render_every == 0 and episode > 0:
                    env.render()
                    time.sleep(0.02)
                
                # Early stopping
                if steps > env.max_steps:
                    break
            
            # Update agent
            loss = agent.update(next_state if not done else None)
            
            # Store metrics
            level_rewards.append(total_reward)
            level_lengths.append(steps)
            agent.episode_rewards.append(total_reward)
            agent.episode_lengths.append(steps)
            
            # Track success
            if total_reward > 50:  # Success threshold
                success_count += 1
            
            # Print progress
            if episode % 100 == 0:
                recent_rewards = level_rewards[-100:] if len(level_rewards) >= 100 else level_rewards
                recent_success = sum(1 for r in recent_rewards if r > 50) / len(recent_rewards)
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(level_lengths[-100:] if len(level_lengths) >= 100 else level_lengths)
                
                print(f"Level {level} Episode {episode}: "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.1f}, "
                      f"Success Rate: {recent_success:.2f}, "
                      f"Loss: {loss:.4f}, "
                      f"Epsilon: {agent.epsilon:.3f}")
            
            # Check if level is mastered
            if episode >= 500 and episode % 100 == 0:
                recent_success_rate = sum(1 for r in level_rewards[-200:] if r > 50) / min(200, len(level_rewards))
                if recent_success_rate > 0.8:  # 80% success rate
                    print(f"🎉 Level {level} mastered! Success rate: {recent_success_rate:.2f}")
                    print(f"Moving to next level after {episode + 1} episodes")
                    break
        
        # Save model after each level
        model_path = f'models/a2c_enhanced_level{level}.pth'
        agent.save_model(model_path)
        
        # Store level results
        all_rewards.extend(level_rewards)
        all_lengths.extend(level_lengths)
        
        env.close()
        
        # Level summary
        final_success_rate = success_count / len(level_rewards)
        print(f"Level {level} completed:")
        print(f"  Episodes: {len(level_rewards)}")
        print(f"  Final Success Rate: {final_success_rate:.2f}")
        print(f"  Average Reward: {np.mean(level_rewards):.2f}")
        print(f"  Average Length: {np.mean(level_lengths):.1f}")
    
    return agent, all_rewards, all_lengths


def test_final_agent(agent, test_mazes, num_episodes=5):
    """Test the final trained agent on all maze levels"""
    
    print("\n=== Testing Final Agent ===")
    
    for maze_name, maze_layout in test_mazes.items():
        print(f"\nTesting on {maze_name}:")
        
        env = MazeEnv(maze_layout, render_mode="human", vision_range=3, use_local_view=True)
        
        test_rewards = []
        test_lengths = []
        
        for episode in range(num_episodes):
            state, info = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            print(f"Test Episode {episode + 1}")
            
            while not done:
                # Use deterministic policy for testing
                action = agent.get_action_with_exploration(state, training=False)
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                steps += 1
                state = next_state
                
                env.render()
                time.sleep(0.1)
                
                if steps > env.max_steps:
                    break
            
            test_rewards.append(total_reward)
            test_lengths.append(steps)
            
            success = "SUCCESS" if total_reward > 50 else "FAILED"
            print(f"Episode {episode + 1}: {success}, Reward: {total_reward:.2f}, Steps: {steps}")
            
            time.sleep(1)
        
        # Test summary for this maze
        avg_reward = np.mean(test_rewards)
        avg_length = np.mean(test_lengths)
        success_rate = sum(1 for r in test_rewards if r > 50) / len(test_rewards)
        
        print(f"{maze_name} Test Results:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Success Rate: {success_rate:.2f}")
        
        env.close()


def main():
    """Main training function"""
    
    # Get curriculum mazes
    curriculum_mazes = create_curriculum_mazes()
    
    print("Available curriculum levels:")
    for i, (name, maze) in enumerate(curriculum_mazes.items(), 1):
        print(f"  Level {i}: {name} ({len(maze)}x{len(maze[0])})")
    
    # Ask user for training mode
    print("\nTraining Options:")
    print("1. Full curriculum training (recommended)")
    print("2. Train on specific level only")
    print("3. Test existing model")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == '1':
        # Full curriculum training
        print("Starting full curriculum training...")
        agent, all_rewards, all_lengths = train_with_curriculum(
            curriculum_mazes, 
            episodes_per_level=3000, 
            render_every=300
        )
        
        # Plot training results
        agent.plot_training_metrics()
        
        # Test final agent
        test_final_agent(agent, curriculum_mazes, num_episodes=3)
        
    elif choice == '2':
        # Train on specific level
        print("Available levels:")
        level_names = list(curriculum_mazes.keys())
        for i, name in enumerate(level_names, 1):
            print(f"  {i}: {name}")
        
        level_choice = int(input("Choose level (1-4): ")) - 1
        if 0 <= level_choice < len(level_names):
            level_name = level_names[level_choice]
            maze_layout = curriculum_mazes[level_name]
            
            # Single level training (similar to original train_agent)
            env = MazeEnv(maze_layout, render_mode="human", vision_range=3, use_local_view=True)
            
            agent = A2CAgent(
                input_size=env.observation_space.shape[0],
                hidden_size=256,
                action_size=env.action_space.n,
                learning_rate=1e-4,
                gamma=0.99,
                value_coef=0.5,
                entropy_coef=0.02,
                use_experience_replay=True
            )
            
            # Training loop (simplified)
            for episode in range(5000):
                state, info = env.reset()
                total_reward = 0
                steps = 0
                done = False
                
                while not done:
                    action = agent.get_action_with_exploration(state, training=True)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    agent.store_transition(action, reward, done)
                    
                    total_reward += reward
                    steps += 1
                    state = next_state
                    
                    if episode % 100 == 0 and episode > 0:
                        env.render()
                        time.sleep(0.05)
                
                loss = agent.update(next_state if not done else None)
                agent.episode_rewards.append(total_reward)
                agent.episode_lengths.append(steps)
                
                if episode % 100 == 0:
                    recent_rewards = agent.episode_rewards[-100:]
                    avg_reward = np.mean(recent_rewards)
                    success_rate = sum(1 for r in recent_rewards if r > 50) / len(recent_rewards)
                    print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")
            
            # Save and test
            agent.save_model(f'models/a2c_enhanced_{level_name}.pth')
            agent.plot_training_metrics()
            env.close()
        
    elif choice == '3':
        # Test existing model
        model_files = [f for f in os.listdir('models') if f.startswith('a2c_enhanced') and f.endswith('.pth')]
        if model_files:
            print("Available models:")
            for i, model in enumerate(model_files, 1):
                print(f"  {i}: {model}")
            
            model_choice = int(input("Choose model: ")) - 1
            if 0 <= model_choice < len(model_files):
                model_path = os.path.join('models', model_files[model_choice])
                
                # Load and test
                env = MazeEnv(list(curriculum_mazes.values())[0], vision_range=3, use_local_view=True)
                agent = A2CAgent(
                    input_size=env.observation_space.shape[0],
                    hidden_size=256,
                    action_size=env.action_space.n
                )
                agent.load_model(model_path)
                env.close()
                
                test_final_agent(agent, curriculum_mazes, num_episodes=3)
        else:
            print("No trained models found!")
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
