import numpy as np

class CloudGamingEnvironment:
    def __init__(self, episode_length=3000):
        self.episode_length = episode_length
        self.current_step = 0
        self.state = None
        self.episode_stats = {
            'latency_violations': 0,
            'total_cost': 0,
            'combat_latencies': []
        }
        
    def reset(self):
        self.current_step = 0
        self.episode_stats = {
            'latency_violations': 0,
            'total_cost': 0,
            'combat_latencies': []
        }
        # Initial state: [combat_prob, latency, network_quality, cpu_load, time_since_combat]
        self.state = np.array([0.1, 0.3, 0.8, 0.5, 1.0])
        return self.state
    
    def step(self, action):
        self.current_step += 1
        
        # Simulate environment response
        slice_latencies = {0: 80, 1: 50, 2: 30}
        slice_costs = {0: 0.1, 1: 0.3, 2: 0.6}
        
        actual_latency = slice_latencies[action] + np.random.normal(0, 5)
        self.episode_stats['total_cost'] += slice_costs[action]
        
        # Calculate reward
        if actual_latency <= 50:
            reward = 1.0
        elif actual_latency <= 100:
            reward = 0.0
        else:
            reward = -1.0
            self.episode_stats['latency_violations'] += 1
        
        # Update state
        self.state = np.random.rand(5)  # Simplified for now
        
        done = self.current_step >= self.episode_length
        
        info = {
            'slice': ['Basic', 'Medium', 'Premium'][action],
            'latency': actual_latency,
            'actual_combat': np.random.random() > 0.7
        }
        
        return self.state, reward, done, info
    
    def get_episode_summary(self):
        return {
            'latency_violations': self.episode_stats['latency_violations'],
            'total_cost': self.episode_stats['total_cost'],
            'avg_combat_latency': np.mean(self.episode_stats['combat_latencies']) if self.episode_stats['combat_latencies'] else 0
        }