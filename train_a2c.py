import numpy as np
import matplotlib.pyplot as plt
import time
import os
from gym_custom import MazeEnv
from agent import A2CAgent


def create_maze_layouts():
    """Create different maze layouts for training and testing"""
    
    # Simple straight maze
    maze_25x25 = [
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
    # Complex maze
    maze_10x10 = [
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,0,0,0,0,0,3,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,0,1,1,1,1,0,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,0,1,1,1,1,1],
        [1,1,0,0,0,0,0,2,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1]
    ]
    
    return {
        'hard': maze_25x25,
        'easy': maze_10x10
    }


def train_agent(maze_layout, num_episodes=1000, render_every=100, use_enhanced_features=True):
    """Train the enhanced A2C agent on a maze"""
    
    # Create environment with enhanced features
    env = MazeEnv(maze_layout, render_mode="human", vision_range=3, use_local_view=use_enhanced_features)
    
    # Create enhanced agent
    input_size = env.observation_space.shape[0]
    hidden_size = 256  # Increased hidden size
    action_size = env.action_space.n  # 4 (up, right, down, left)
    
    print(f"Input size: {input_size}, Hidden size: {hidden_size}, Action size: {action_size}")
    
    agent = A2CAgent(
        input_size=input_size,
        hidden_size=hidden_size,
        action_size=action_size,
        learning_rate=1e-4,  # Lower learning rate for stability
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.02,  # Higher entropy for more exploration
        use_lstm=False,  # Can be enabled for memory
        use_experience_replay=True,
        replay_buffer_size=10000,
        batch_size=32
    )
    
    print(f"Training A2C agent for {num_episodes} episodes...")
    print(f"Environment: {env.height}x{env.width} maze")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    
    # Training loop
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Get action from agent with exploration
            action = agent.get_action_with_exploration(state, training=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(action, reward, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            # Render occasionally during training
            if episode % render_every == 0 and episode > 0:
                env.render()
                time.sleep(0.05)
                
            # Early stopping if episode is too long
            if steps > env.max_steps:
                break
        
        # Update agent after episode
        loss = agent.update(next_state if not done else None)
        
        # Store episode metrics
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        
        # Print progress
        if episode % 50 == 0:
            avg_reward = np.mean(agent.episode_rewards[-50:]) if len(agent.episode_rewards) >= 50 else np.mean(agent.episode_rewards)
            avg_length = np.mean(agent.episode_lengths[-50:]) if len(agent.episode_lengths) >= 50 else np.mean(agent.episode_lengths)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, Loss: {loss:.4f}")
            
            # Check if solved (high success rate)
            if len(agent.episode_rewards) >= 100:
                recent_rewards = agent.episode_rewards[-100:]
                success_rate = sum(1 for r in recent_rewards if r > 50) / len(recent_rewards)
                if success_rate > 0.8:
                    print(f"Maze solved! Success rate: {success_rate:.2f}")
                    break
    
    env.close()
    return agent


def test_agent(agent, maze_layout, num_episodes=5, render=True):
    """Test the trained agent"""
    
    env = MazeEnv(maze_layout, render_mode="human" if render else None)
    
    print(f"\nTesting agent for {num_episodes} episodes...")
    
    test_rewards = []
    test_lengths = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        print(f"\nTest Episode {episode + 1}")
        print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
        
        while not done:
            # Get action from agent (no exploration)
            action = agent.get_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if render:
                env.render()
                time.sleep(0.1)
            
            # Safety check
            if steps > 500:
                print("Episode too long, stopping...")
                break
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        
        success = "SUCCESS" if total_reward > 50 else "FAILED"
        print(f"Episode {episode + 1}: {success}, Reward: {total_reward:.2f}, Steps: {steps}")
        print(f"Path length: {len(env.path)}, Visited positions: {len(env.visited_positions)}")
        
        if render:
            time.sleep(2)  # Pause between episodes
    
    env.close()
    
    # Print test summary
    avg_reward = np.mean(test_rewards)
    avg_length = np.mean(test_lengths)
    success_rate = sum(1 for r in test_rewards if r > 50) / len(test_rewards)
    
    print(f"\nTest Summary:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Length: {avg_length:.1f}")
    print(f"Success Rate: {success_rate:.2f}")
    
    return test_rewards, test_lengths


def main():
    """Main function to run training and testing"""
    
    # Get maze layouts
    mazes = create_maze_layouts()
    
    # Choose maze for training
    maze_name = 'hard'  # Change this to 'straight', 'l_shaped', or 'complex'
    maze_layout = mazes[maze_name]
    
    print(f"Using {maze_name} maze layout")
    print("Maze layout:")
    for row in maze_layout:
        print(''.join(['█' if cell == 1 else ' ' for cell in row]))
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'a2c_maze_hard.pth')
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Train new agent")
    print("2. Load and test existing agent")
    print("3. Train and then test")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == '1':
        # Train new agent
        agent = train_agent(maze_layout, num_episodes=5000, render_every=1)
        agent.save_model(model_path)
        agent.plot_training_metrics()
        
    elif choice == '2':
        # Load and test existing agent
        if os.path.exists(model_path):
            # Create agent with same architecture
            env = MazeEnv(maze_layout)
            agent = A2CAgent(
                input_size=env.observation_space.shape[0],
                hidden_size=128,
                action_size=env.action_space.n
            )
            agent.load_model(model_path)
            env.close()
            
            # Test the agent
            test_agent(agent, maze_layout, num_episodes=5, render=True)
        else:
            print(f"No trained model found at {model_path}")
            
    elif choice == '3':
        # Train and test
        agent = train_agent(maze_layout, num_episodes=10000, render_every=100)
        agent.save_model(model_path)
        
        print("\nTraining completed! Now testing the agent...")
        time.sleep(2)
        
        test_agent(agent, maze_layout, num_episodes=3, render=True)
        agent.plot_training_metrics()
        
    else:
        print("Invalid choice!")

 
if __name__ == "__main__":
    main()
