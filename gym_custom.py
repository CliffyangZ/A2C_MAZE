import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gymnasium import spaces
from typing import Optional, List, Tuple
import torch


class MazeEnv(gym.Env):
    """Custom Maze Environment with path tracking for RL training"""
    
    def __init__(self, maze_map: List[List[int]], render_mode: Optional[str] = None, 
                 vision_range: int = 3, use_local_view: bool = True):
        super().__init__()
        
        self.maze_map = np.array(maze_map)
        self.height, self.width = self.maze_map.shape
        self.render_mode = render_mode
        self.vision_range = vision_range
        self.use_local_view = use_local_view
        
        # Find start and goal positions
        self.start_pos = self._find_free_position()
        self.goal_pos = self._find_goal_position()
        
        # Current agent position
        self.agent_pos = self.start_pos.copy()
        
        # Path tracking
        self.path = [self.start_pos.copy()]
        self.visited_positions = set()
        self.visited_positions.add(tuple(self.start_pos))
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Enhanced observation space
        if self.use_local_view:
            # Local view + relative goal position + additional features
            local_view_size = (2 * vision_range + 1) ** 2
            obs_size = local_view_size + 6  # local view + goal_rel_x + goal_rel_y + distance + visited_ratio + step_ratio + last_action
        else:
            # Original simple observation
            obs_size = 4
            
        self.observation_space = spaces.Box(
            low=-1, high=1, 
            shape=(obs_size,), dtype=np.float32
        )
        
        # Rendering
        self.fig = None
        self.ax = None
        
        # Episode tracking
        self.max_steps = 300  # Increased for more complex mazes
        self.current_step = 0
        self.last_action = -1  # Track last action for observation
        
    def _find_free_position(self) -> np.ndarray:
        """Find a start position: prefer cells marked 3, else first free (0)."""
        # Prefer explicit start marker (3)
        start_positions = np.where(self.maze_map == 3)
        if len(start_positions[0]) > 0:
            idx = 0  # Use first 3 as start
            return np.array([start_positions[0][idx], start_positions[1][idx]])

        # Fallback to first free cell (0)
        free_positions = np.where(self.maze_map == 0)
        if len(free_positions[0]) > 0:
            idx = 0  # Use first free position as start
            return np.array([free_positions[0][idx], free_positions[1][idx]])
        return np.array([1, 1])  # Default position
    
    def _find_goal_position(self) -> np.ndarray:
        """Find goal position: prefer cells marked 2, else last free (0)."""
        # Prefer explicit goal marker (2)
        goal_positions = np.where(self.maze_map == 2)
        if len(goal_positions[0]) > 0:
            idx = 0  # Use first 2 as goal
            return np.array([goal_positions[0][idx], goal_positions[1][idx]])

        # Fallback to last free cell (0)
        free_positions = np.where(self.maze_map == 0)
        if len(free_positions[0]) > 0:
            idx = -1  # Use last free position as goal
            return np.array([free_positions[0][idx], free_positions[1][idx]])
        return np.array([self.height-2, self.width-2])  # Default goal
    
    def _get_local_view(self) -> np.ndarray:
        """Get local view around the agent"""
        local_view = []
        agent_row, agent_col = self.agent_pos
        
        for dr in range(-self.vision_range, self.vision_range + 1):
            for dc in range(-self.vision_range, self.vision_range + 1):
                r, c = agent_row + dr, agent_col + dc
                
                if 0 <= r < self.height and 0 <= c < self.width:
                    cell_value = self.maze_map[r, c]
                    # Normalize cell values: wall=1, free=0, goal=0.5, start=0.3
                    if cell_value == 1:  # Wall
                        local_view.append(1.0)
                    elif cell_value == 2:  # Goal
                        local_view.append(0.5)
                    elif cell_value == 3:  # Start
                        local_view.append(0.3)
                    else:  # Free space
                        # Check if visited
                        if (r, c) in self.visited_positions:
                            local_view.append(-0.3)  # Visited free space
                        else:
                            local_view.append(0.0)  # Unvisited free space
                else:
                    # Out of bounds treated as wall
                    local_view.append(1.0)
        
        return np.array(local_view, dtype=np.float32)
    
    def _get_observation(self) -> np.ndarray:
        """Get enhanced observation with local view and additional features"""
        if not self.use_local_view:
            # Original simple observation
            return np.array([
                self.agent_pos[0] / self.height,
                self.agent_pos[1] / self.width,
                self.goal_pos[0] / self.height,
                self.goal_pos[1] / self.width
            ], dtype=np.float32)
        
        # Enhanced observation with local view
        local_view = self._get_local_view()
        
        # Relative goal position (normalized)
        goal_rel_x = (self.goal_pos[1] - self.agent_pos[1]) / self.width
        goal_rel_y = (self.goal_pos[0] - self.agent_pos[0]) / self.height
        
        # Distance to goal (normalized)
        distance = np.linalg.norm(self.agent_pos - self.goal_pos) / np.sqrt(self.height**2 + self.width**2)
        
        # Visited ratio (exploration progress)
        total_free_cells = np.sum((self.maze_map == 0) | (self.maze_map == 2) | (self.maze_map == 3))
        visited_ratio = len(self.visited_positions) / max(total_free_cells, 1)
        
        # Step ratio (time pressure)
        step_ratio = self.current_step / self.max_steps
        
        # Last action (normalized)
        last_action_norm = self.last_action / 3.0 if self.last_action >= 0 else 0.0
        
        # Combine all features
        observation = np.concatenate([
            local_view,
            [goal_rel_x, goal_rel_y, distance, visited_ratio, step_ratio, last_action_norm]
        ])
        
        return observation.astype(np.float32)
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is valid (within bounds and not a wall)"""
        row, col = pos
        if 0 <= row < self.height and 0 <= col < self.width:
            # Walkable cells: 0 (free), 2 (goal), 3 (start)
            return self.maze_map[row, col] in (0, 2, 3)
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        prev_pos = self.agent_pos.copy()
        
        # Define action mappings
        action_map = {
            0: [-1, 0],  # up
            1: [0, 1],   # right
            2: [1, 0],   # down
            3: [0, -1]   # left
        }
        
        # Calculate new position
        new_pos = self.agent_pos + np.array(action_map[action])
        
        # Check if new position is valid
        moved = False
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
            self.path.append(self.agent_pos.copy())
            moved = True
            
            # Track visited positions for reward shaping
            pos_tuple = tuple(self.agent_pos)
            exploration_bonus = 0.2 if pos_tuple not in self.visited_positions else -0.05  # Penalty for revisiting
            self.visited_positions.add(pos_tuple)
        else:
            exploration_bonus = -0.1  # Penalty for hitting wall
        
        # Update last action
        self.last_action = action
        
        # Calculate reward
        reward = self._calculate_reward(exploration_bonus, moved, prev_pos, action)
        
        # Check if goal is reached
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        
        # Check if max steps reached
        truncated = self.current_step >= self.max_steps
        
        info = {
            'agent_pos': self.agent_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'path_length': len(self.path),
            'visited_count': len(self.visited_positions),
            'moved': moved
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _calculate_reward(self, exploration_bonus: float, moved: bool, prev_pos: np.ndarray, action: int) -> float:
        """Enhanced reward function with multiple components"""
        
        # Goal reached reward (highest priority)
        if np.array_equal(self.agent_pos, self.goal_pos):
            # Bonus for reaching goal efficiently
            efficiency_bonus = max(0, (self.max_steps - self.current_step) / self.max_steps * 50)
            return 100.0 + efficiency_bonus
        
        # Distance-based reward (progress towards goal)
        current_distance = np.linalg.norm(self.agent_pos - self.goal_pos)
        prev_distance = np.linalg.norm(prev_pos - self.goal_pos)
        distance_reward = (prev_distance - current_distance) * 2.0  # Reward for getting closer
        
        # Movement reward/penalty
        movement_reward = 0.05 if moved else -0.2  # Penalty for not moving (hitting walls)
        
        # Step penalty (encourage efficiency)
        step_penalty = -0.02
        
        # Backtracking penalty (discourage going back to previous position)
        backtrack_penalty = 0
        if len(self.path) >= 3 and moved:
            if np.array_equal(self.agent_pos, self.path[-3]):
                backtrack_penalty = -0.3
        
        # Progress reward (getting closer to unexplored areas near goal)
        progress_reward = 0
        if moved:
            # Reward for exploring areas closer to goal
            goal_distance_factor = 1.0 / (1.0 + current_distance)
            progress_reward = exploration_bonus * goal_distance_factor
        
        # Time pressure (increase urgency as steps increase)
        time_pressure = -0.01 * (self.current_step / self.max_steps)
        
        # Total reward
        total_reward = (distance_reward + movement_reward + step_penalty + 
                       backtrack_penalty + progress_reward + time_pressure)
        
        return total_reward
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.agent_pos = self.start_pos.copy()
        self.path = [self.start_pos.copy()]
        self.visited_positions = set()
        self.visited_positions.add(tuple(self.start_pos))
        self.current_step = 0
        self.last_action = -1  # Reset last action
        
        info = {
            'agent_pos': self.agent_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'path_length': len(self.path),
            'visited_count': len(self.visited_positions)
        }
        
        return self._get_observation(), info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                plt.ion()
            
            self.ax.clear()
            
            # Draw maze
            for i in range(self.height):
                for j in range(self.width):
                    if self.maze_map[i, j] == 1:  # Wall
                        rect = patches.Rectangle((j, self.height-1-i), 1, 1, 
                                               linewidth=1, edgecolor='black', 
                                               facecolor='black')
                        self.ax.add_patch(rect)
                    else:  # Free space
                        rect = patches.Rectangle((j, self.height-1-i), 1, 1, 
                                               linewidth=1, edgecolor='gray', 
                                               facecolor='white')
                        self.ax.add_patch(rect)
            
            # Draw path
            if len(self.path) > 1:
                path_array = np.array(self.path)
                # Convert to display coordinates
                path_x = path_array[:, 1] + 0.5
                path_y = self.height - 1 - path_array[:, 0] + 0.5
                self.ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Path')
                
                # Draw path points
                self.ax.scatter(path_x[:-1], path_y[:-1], c='lightblue', s=30, alpha=0.6)
            
            # Draw start position
            start_x, start_y = self.start_pos[1] + 0.5, self.height - 1 - self.start_pos[0] + 0.5
            self.ax.scatter(start_x, start_y, c='green', s=100, marker='s', label='Start')
            
            # Draw goal position
            goal_x, goal_y = self.goal_pos[1] + 0.5, self.height - 1 - self.goal_pos[0] + 0.5
            self.ax.scatter(goal_x, goal_y, c='red', s=100, marker='*', label='Goal')
            
            # Draw current agent position
            agent_x, agent_y = self.agent_pos[1] + 0.5, self.height - 1 - self.agent_pos[0] + 0.5
            self.ax.scatter(agent_x, agent_y, c='blue', s=150, marker='o', label='Agent')
            
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')
            self.ax.set_title(f'Maze Environment - Step: {self.current_step}, Path Length: {len(self.path)}')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            plt.pause(0.01)
            plt.draw()
    
    def close(self):
        """Close the rendering"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

            