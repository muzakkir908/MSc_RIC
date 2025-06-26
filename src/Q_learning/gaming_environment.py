import numpy as np
import pandas as pd
from collections import deque
import json

class CloudGamingEnvironment:
    """
    Simulates cloud gaming environment for Q-learning
    
    State: [combat_prob, current_latency, network_quality, cpu_load, time_since_combat]
    Actions: 0 = Basic slice, 1 = Medium slice, 2 = Premium slice
    """
    
    def __init__(self, episode_length=3000):  # 5 minutes at 10Hz
        self.episode_length = episode_length
        self.current_step = 0
        
        # Network slice properties
        self.slices = {
            0: {'name': 'Basic', 'base_latency': 80, 'cost': 0.1, 'bandwidth': 50},
            1: {'name': 'Medium', 'base_latency': 50, 'cost': 0.3, 'bandwidth': 100},
            2: {'name': 'Premium', 'base_latency': 30, 'cost': 0.6, 'bandwidth': 200}
        }
        
        # Current state variables
        self.current_slice = 0
        self.combat_probability = 0.0
        self.actual_combat = False
        self.time_since_combat = 0
        self.network_congestion = 0.0
        
        # Performance tracking
        self.episode_stats = {
            'total_cost': 0,
            'latency_violations': 0,
            'combat_latency_sum': 0,
            'combat_timesteps': 0,
            'slice_usage': [0, 0, 0]
        }
        
        # Combat phases for realistic scenarios
        self.combat_schedule = self._generate_combat_schedule()
        
    def _generate_combat_schedule(self):
        """Generate realistic combat phases"""
        schedule = []
        t = 0
        
        while t < self.episode_length:
            # Peaceful period (30-120 seconds)
            peaceful_duration = np.random.randint(300, 1200)
            t += peaceful_duration
            
            if t < self.episode_length:
                # Combat period (10-60 seconds)
                combat_duration = np.random.randint(100, 600)
                combat_start = t
                combat_end = min(t + combat_duration, self.episode_length)
                schedule.append((combat_start, combat_end))
                t = combat_end
        
        return schedule
    
    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.current_slice = 0
        self.combat_probability = 0.0
        self.actual_combat = False
        self.time_since_combat = 0
        self.network_congestion = np.random.uniform(0, 0.3)
        
        # Reset stats
        self.episode_stats = {
            'total_cost': 0,
            'latency_violations': 0,
            'combat_latency_sum': 0,
            'combat_timesteps': 0,
            'slice_usage': [0, 0, 0]
        }
        
        # Generate new combat schedule
        self.combat_schedule = self._generate_combat_schedule()
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state observation"""
        # Normalize state values to [0, 1]
        state = np.array([
            self.combat_probability,                    # 0-1
            min(self._get_current_latency() / 150, 1), # Normalized latency
            1 - self.network_congestion,                # Network quality (0-1)
            np.random.uniform(0.3, 0.8),               # CPU load
            min(self.time_since_combat / 300, 1)       # Time since last combat
        ])
        return state
    
    def _get_current_latency(self):
        """Calculate current latency based on slice and conditions"""
        base_latency = self.slices[self.current_slice]['base_latency']
        
        # Add network congestion effect
        congestion_factor = 1 + self.network_congestion
        
        # Add random variation
        variation = np.random.uniform(0.8, 1.2)
        
        # Combat increases latency slightly
        combat_factor = 1.2 if self.actual_combat else 1.0
        
        latency = base_latency * congestion_factor * variation * combat_factor
        return latency
    
    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        # Validate action
        assert action in [0, 1, 2], f"Invalid action: {action}"
        
        # Update slice if changed
        self.current_slice = action
        self.episode_stats['slice_usage'][action] += 1
        
        # Update combat state
        self._update_combat_state()
        
        # Calculate current latency
        current_latency = self._get_current_latency()
        
        # Calculate reward
        reward = self._calculate_reward(action, current_latency)
        
        # Update statistics
        self.episode_stats['total_cost'] += self.slices[action]['cost']
        if current_latency > 100:
            self.episode_stats['latency_violations'] += 1
        if self.actual_combat:
            self.episode_stats['combat_timesteps'] += 1
            self.episode_stats['combat_latency_sum'] += current_latency
        
        # Move to next timestep
        self.current_step += 1
        self.time_since_combat = 0 if self.actual_combat else self.time_since_combat + 1
        
        # Check if episode is done
        done = self.current_step >= self.episode_length
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'latency': current_latency,
            'actual_combat': self.actual_combat,
            'slice': self.slices[action]['name']
        }
        
        return next_state, reward, done, info
    
    def _update_combat_state(self):
        """Update combat probability and actual combat state"""
        # Check if we're in a combat phase
        self.actual_combat = any(
            start <= self.current_step < end 
            for start, end in self.combat_schedule
        )
        
        # Simulate LSTM predictions (with some error)
        if self.actual_combat:
            # During combat, high probability with some noise
            self.combat_probability = np.clip(np.random.normal(0.9, 0.1), 0, 1)
        else:
            # Check if combat is approaching
            time_to_combat = min(
                start - self.current_step 
                for start, end in self.combat_schedule 
                if start > self.current_step
            ) if any(start > self.current_step for start, _ in self.combat_schedule) else 1000
            
            if time_to_combat < 30:  # 3 seconds before combat
                # Probability increases as combat approaches
                self.combat_probability = np.clip(1 - time_to_combat / 30, 0, 1)
            else:
                # Low probability with noise
                self.combat_probability = np.clip(np.random.normal(0.1, 0.05), 0, 1)
        
        # Update network congestion
        self.network_congestion = np.clip(
            self.network_congestion + np.random.normal(0, 0.01), 0, 0.5
        )
    
    def _calculate_reward(self, action, latency):
        """Calculate reward based on action and resulting latency"""
        reward = 0
        
        # Latency-based reward/penalty
        if latency <= 50:
            reward += 1.0  # Excellent
        elif latency <= 80:
            reward += 0.5  # Good
        elif latency <= 100:
            reward += 0.0  # Acceptable
        else:
            reward -= 1.0  # Poor
        
        # Combat-specific rewards/penalties
        if self.actual_combat:
            if action == 2:  # Premium during combat
                reward += 0.5  # Good choice
            elif action == 0:  # Basic during combat
                reward -= 1.0  # Bad choice
        else:
            # Penalize unnecessary premium usage
            if action == 2:
                reward -= 0.3  # Waste
            elif action == 0:
                reward += 0.1  # Efficient
        
        # Small penalty for slice switching (to encourage stability)
        # This will be added in the Q-learning agent
        
        return reward
    
    def get_episode_summary(self):
        """Get summary statistics for the episode"""
        avg_combat_latency = (
            self.episode_stats['combat_latency_sum'] / self.episode_stats['combat_timesteps']
            if self.episode_stats['combat_timesteps'] > 0 else 0
        )
        
        return {
            'total_cost': self.episode_stats['total_cost'],
            'latency_violations': self.episode_stats['latency_violations'],
            'avg_combat_latency': avg_combat_latency,
            'slice_distribution': [
                usage / sum(self.episode_stats['slice_usage']) 
                for usage in self.episode_stats['slice_usage']
            ],
            'violation_rate': self.episode_stats['latency_violations'] / self.episode_length
        }

# Test the environment
if __name__ == "__main__":
    env = CloudGamingEnvironment()
    state = env.reset()
    
    print("🎮 Testing Cloud Gaming Environment")
    print("=" * 50)
    print(f"Initial state: {state}")
    print(f"State size: {len(state)}")
    print(f"Action space: 0=Basic, 1=Medium, 2=Premium")
    
    # Run a few steps
    for i in range(10):
        action = np.random.choice([0, 1, 2])  # Random action
        next_state, reward, done, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Action: {info['slice']}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Latency: {info['latency']:.1f}ms")
        print(f"  Combat: {info['actual_combat']}")