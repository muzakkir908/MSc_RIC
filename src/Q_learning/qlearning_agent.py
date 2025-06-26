import numpy as np
import pickle
import json
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearningAgent:
    """
    Q-Learning agent for network slice selection
    Uses discretized states for table-based Q-learning
    """
    
    def __init__(self, state_size=5, action_size=3, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table using defaultdict for automatic initialization
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # State discretization parameters
        self.state_bins = [
            np.array([0.0, 0.3, 0.7, 1.0]),      # Combat probability
            np.array([0.0, 0.33, 0.67, 1.0]),    # Latency (normalized)
            np.array([0.0, 0.5, 1.0]),           # Network quality
            np.array([0.0, 0.5, 1.0]),           # CPU load
            np.array([0.0, 0.33, 0.67, 1.0])     # Time since combat
        ]
        
        # Performance tracking
        self.training_history = {
            'episode_rewards': [],
            'episode_costs': [],
            'episode_violations': [],
            'epsilon_values': [],
            'avg_q_values': []
        }
        
        # Action tracking
        self.last_action = None
    
    def discretize_state(self, state):
        """Convert continuous state to discrete for Q-table lookup"""
        discrete_state = []
        for i, value in enumerate(state):
            # Find which bin the value falls into
            bin_index = np.digitize(value, self.state_bins[i]) - 1
            bin_index = np.clip(bin_index, 0, len(self.state_bins[i]) - 2)
            discrete_state.append(bin_index)
        return tuple(discrete_state)
    
    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        # Discretize the state
        discrete_state = self.discretize_state(state)
        
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.choice(self.action_size)
        else:
            # Exploitation: best action from Q-table
            q_values = self.q_table[discrete_state]
            action = np.argmax(q_values)
        
        self.last_action = action
        return action
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning formula"""
        # Discretize states
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Add penalty for switching slices (encourages stability)
        if self.last_action is not None and action != self.last_action:
            reward -= 0.1
        
        # Q-learning update
        current_q = self.q_table[discrete_state][action]
        
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[discrete_next_state])
            target_q = reward + self.discount_factor * next_max_q
        
        # Update Q-value
        self.q_table[discrete_state][action] = (
            (1 - self.learning_rate) * current_q + 
            self.learning_rate * target_q
        )
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath='q_learning_model.pkl'):
        """Save the trained Q-table and parameters"""
        model_data = {
            'q_table': dict(self.q_table),
            'state_bins': self.state_bins,
            'training_history': self.training_history,
            'parameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save a JSON summary
        summary = {
            'total_states_explored': len(self.q_table),
            'final_epsilon': self.epsilon,
            'episodes_trained': len(self.training_history['episode_rewards']),
            'average_final_reward': np.mean(self.training_history['episode_rewards'][-10:]) if self.training_history['episode_rewards'] else 0
        }
        
        with open(filepath.replace('.pkl', '_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath='q_learning_model.pkl'):
        """Load a trained Q-table"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), model_data['q_table'])
        self.state_bins = model_data['state_bins']
        self.training_history = model_data['training_history']
        
        params = model_data['parameters']
        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.learning_rate = params['learning_rate']
        self.discount_factor = params['discount_factor']
        self.epsilon = params['epsilon']
        
        print(f"✅ Model loaded from {filepath}")
        print(f"   States explored: {len(self.q_table)}")
        print(f"   Episodes trained: {len(self.training_history['episode_rewards'])}")
    
    def get_policy_summary(self):
        """Analyze the learned policy"""
        policy_analysis = {
            'low_combat': {'basic': 0, 'medium': 0, 'premium': 0},
            'medium_combat': {'basic': 0, 'medium': 0, 'premium': 0},
            'high_combat': {'basic': 0, 'medium': 0, 'premium': 0}
        }
        
        action_names = ['basic', 'medium', 'premium']
        
        for state, q_values in self.q_table.items():
            combat_prob_bin = state[0]
            best_action = np.argmax(q_values)
            
            if combat_prob_bin == 0:  # Low combat probability
                policy_analysis['low_combat'][action_names[best_action]] += 1
            elif combat_prob_bin == 1:  # Medium combat probability
                policy_analysis['medium_combat'][action_names[best_action]] += 1
            else:  # High combat probability
                policy_analysis['high_combat'][action_names[best_action]] += 1
        
        return policy_analysis


class QLearningTrainer:
    """Handles the training process for the Q-learning agent"""
    
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        
    def train(self, episodes=1000, verbose=True):
        """Train the Q-learning agent"""
        print(f"🚀 Starting Q-Learning Training for {episodes} episodes")
        print("=" * 60)
        
        for episode in range(episodes):
            # Reset environment
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            # Episode loop
            while True:
                # Choose action
                action = self.agent.get_action(state, training=True)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Learn from experience
                self.agent.learn(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Get episode summary
            episode_summary = self.env.get_episode_summary()
            
            # Update training history
            self.agent.training_history['episode_rewards'].append(total_reward)
            self.agent.training_history['episode_costs'].append(episode_summary['total_cost'])
            self.agent.training_history['episode_violations'].append(episode_summary['latency_violations'])
            self.agent.training_history['epsilon_values'].append(self.agent.epsilon)
            
            # Calculate average Q-value for monitoring
            avg_q = np.mean([np.max(q_values) for q_values in self.agent.q_table.values()]) if self.agent.q_table else 0
            self.agent.training_history['avg_q_values'].append(avg_q)
            
            # Progress update
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.agent.training_history['episode_rewards'][-100:])
                avg_violations = np.mean(self.agent.training_history['episode_violations'][-100:])
                print(f"Episode {episode + 1}/{episodes}")
                print(f"  Avg Reward (last 100): {avg_reward:.2f}")
                print(f"  Avg Violations (last 100): {avg_violations:.1f}")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                print(f"  States explored: {len(self.agent.q_table)}")
        
        print("\n✅ Training Complete!")
        
        # Analyze learned policy
        policy = self.agent.get_policy_summary()
        print("\n📊 Learned Policy Summary:")
        for combat_level, actions in policy.items():
            print(f"\n{combat_level.replace('_', ' ').title()}:")
            total = sum(actions.values())
            if total > 0:
                for action, count in actions.items():
                    print(f"  {action.title()}: {count/total*100:.1f}%")
    
    def plot_training_results(self):
        """Visualize training progress"""
        history = self.agent.training_history
        
        if not history['episode_rewards']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Smooth curves for better visualization
        def smooth(values, window=50):
            if len(values) < window:
                return values
            return np.convolve(values, np.ones(window)/window, mode='valid')
        
        # Plot 1: Episode Rewards
        ax = axes[0, 0]
        episodes = range(len(history['episode_rewards']))
        ax.plot(episodes, history['episode_rewards'], alpha=0.3, color='blue')
        if len(history['episode_rewards']) > 50:
            ax.plot(range(49, len(history['episode_rewards'])), 
                   smooth(history['episode_rewards']), 
                   color='blue', linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Rewards Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Latency Violations
        ax = axes[0, 1]
        ax.plot(episodes, history['episode_violations'], alpha=0.3, color='red')
        if len(history['episode_violations']) > 50:
            ax.plot(range(49, len(history['episode_violations'])), 
                   smooth(history['episode_violations']), 
                   color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Violations per Episode')
        ax.set_title('Latency Violations Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Epsilon Decay
        ax = axes[1, 0]
        ax.plot(episodes, history['epsilon_values'], color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon (Exploration Rate)')
        ax.set_title('Exploration vs Exploitation')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Average Q-Values
        ax = axes[1, 1]
        ax.plot(episodes, history['avg_q_values'], color='purple')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Q-Value')
        ax.set_title('Q-Value Growth Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('q_learning_training_results.png', dpi=150)
        plt.show()
        
        print("✅ Training plots saved to 'q_learning_training_results.png'")

# Test the Q-learning agent
if __name__ == "__main__":
    from gaming_environment import CloudGamingEnvironment
    
    # Create environment and agent
    env = CloudGamingEnvironment()
    agent = QLearningAgent()
    
    print("🤖 Q-Learning Agent Initialized")
    print(f"   State bins: {[len(bins)-1 for bins in agent.state_bins]} bins per feature")
    print(f"   Actions: 3 (Basic, Medium, Premium)")
    print(f"   Initial epsilon: {agent.epsilon}")