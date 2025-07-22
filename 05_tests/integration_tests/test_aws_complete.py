#!/usr/bin/env python3
"""
Complete AWS System Test - Full Gaming Simulation
Tests your deployed Edge-Cloud system with realistic scenarios
"""

import requests
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class AWSSystemTest:
    def __init__(self):
        # YOUR AWS IPs - Already deployed
        self.edge_url = "http://ec2-54-173-203-87.compute-1.amazonaws.com:5000"
        self.cloud_url = "http://13.220.117.61:5001"

        
        # Test results storage
        self.results = {
            'timestamps': [],
            'actual_combat': [],
            'predicted_combat': [],
            'combat_probability': [],
            'slice_decision': [],
            'edge_latency': [],
            'cloud_latency': [],
            'total_latency': []
        }
        
    def warm_up_edge_buffer(self):
        """Warm up LSTM buffer with 30 data points"""
        print("Warming up Edge server buffer...")
        
        # Reset buffer first
        requests.post(f"{self.edge_url}/reset")
        
        # Send 30 peaceful data points to fill buffer
        peaceful_state = {
            "mouse_speed": 150,
            "turning_rate": 80,
            "movement_keys": 1,
            "is_shooting": 0,
            "activity_score": 0.2,
            "keys_pressed": 2,
            "ping_ms": 45,
            "cpu_percent": 35,
            "mouse_speed_ma": 140,
            "activity_ma": 0.18,
            "combat_likelihood": 0.1
        }
        
        for i in range(30):
            response = requests.post(f"{self.edge_url}/predict", json=peaceful_state)
            print(f"\rBuffer: {i+1}/30", end="")
        print("\n✅ Buffer ready!")
        
    def generate_game_state(self, in_combat):
        """Generate realistic game state based on combat status"""
        if in_combat:
            return {
                "mouse_speed": np.random.uniform(600, 1000),
                "turning_rate": np.random.uniform(400, 800),
                "movement_keys": np.random.randint(2, 4),
                "is_shooting": int(np.random.choice([0, 1], p=[0.3, 0.7])),
                "activity_score": np.random.uniform(0.7, 0.95),
                "keys_pressed": np.random.randint(4, 7),
                "ping_ms": np.random.uniform(50, 80),
                "cpu_percent": np.random.uniform(60, 85),
                "mouse_speed_ma": np.random.uniform(580, 980),
                "activity_ma": np.random.uniform(0.65, 0.92),
                "combat_likelihood": np.random.uniform(0.75, 0.95)
            }
        else:
            return {
                "mouse_speed": np.random.uniform(50, 300),
                "turning_rate": np.random.uniform(30, 200),
                "movement_keys": np.random.randint(0, 2),
                "is_shooting": 0,
                "activity_score": np.random.uniform(0.1, 0.4),
                "keys_pressed": np.random.randint(1, 3),
                "ping_ms": np.random.uniform(35, 60),
                "cpu_percent": np.random.uniform(30, 50),
                "mouse_speed_ma": np.random.uniform(45, 280),
                "activity_ma": np.random.uniform(0.08, 0.35),
                "combat_likelihood": np.random.uniform(0.05, 0.25)
            }
    
    def run_gaming_simulation(self, duration_seconds=300):
        """Run 5-minute gaming simulation"""
        print(f"\n🎮 RUNNING {duration_seconds}s GAMING SIMULATION")
        print("=" * 60)
        
        # Warm up first
        self.warm_up_edge_buffer()
        
        # Define combat phases (realistic FPS pattern)
        combat_phases = [
            (30, 50),    # Combat at 30-50s
            (90, 120),   # Combat at 90-120s
            (180, 210),  # Combat at 180-210s
            (250, 280),  # Combat at 250-280s
        ]
        
        # Simulation loop
        for t in range(duration_seconds):
            # Check if in combat phase
            in_combat = any(start <= t < end for start, end in combat_phases)
            
            # Generate game state
            game_state = self.generate_game_state(in_combat)
            
            # EDGE: Get combat prediction
            start_edge = time.time()
            edge_response = requests.post(f"{self.edge_url}/predict", json=game_state)
            edge_latency = (time.time() - start_edge) * 1000
            
            if edge_response.status_code == 200:
                prediction = edge_response.json()
                
                # CLOUD: Get slice decision
                cloud_data = {
                    'combat_probability': prediction['probability'],
                    'current_latency': game_state['ping_ms'],
                    'network_quality': 0.8,
                    'cpu_load': game_state['cpu_percent'] / 100,
                    'time_since_combat': 0 if in_combat else t
                }
                
                start_cloud = time.time()
                cloud_response = requests.post(f"{self.cloud_url}/decide", json=cloud_data)
                cloud_latency = (time.time() - start_cloud) * 1000
                
                if cloud_response.status_code == 200:
                    decision = cloud_response.json()
                    
                    # Store results
                    self.results['timestamps'].append(t)
                    self.results['actual_combat'].append(in_combat)
                    self.results['predicted_combat'].append(prediction['prediction'])
                    self.results['combat_probability'].append(prediction['probability'])
                    self.results['slice_decision'].append(decision['action'])
                    self.results['edge_latency'].append(edge_latency)
                    self.results['cloud_latency'].append(cloud_latency)
                    self.results['total_latency'].append(edge_latency + cloud_latency)
                    
                    # Progress update every 10 seconds
                    if t % 10 == 0:
                        status = "⚔️ COMBAT" if in_combat else "🚶 Peaceful"
                        pred_status = "🎯 Combat predicted" if prediction['prediction'] else "✅ Peaceful"
                        print(f"[{t:3d}s] {status} | {pred_status} ({prediction['probability']:.0%}) | "
                              f"Slice: {decision['slice']} | Latency: {edge_latency:.1f}+{cloud_latency:.1f}ms")
            
            # Small delay to not overwhelm servers
            time.sleep(0.1)
        
        print("\n✅ Simulation complete!")
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze and visualize results"""
        print("\n📊 ANALYZING RESULTS...")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Calculate metrics
        metrics = {
            'total_predictions': len(df),
            'correct_predictions': sum(df['actual_combat'] == df['predicted_combat']),
            'accuracy': sum(df['actual_combat'] == df['predicted_combat']) / len(df) * 100,
            'avg_edge_latency': df['edge_latency'].mean(),
            'avg_cloud_latency': df['cloud_latency'].mean(),
            'avg_total_latency': df['total_latency'].mean(),
            'latency_under_50ms': sum(df['total_latency'] < 50) / len(df) * 100,
            'slice_distribution': df['slice_decision'].value_counts().to_dict()
        }
        
        # Combat vs Non-combat analysis
        combat_df = df[df['actual_combat'] == True]
        peaceful_df = df[df['actual_combat'] == False]
        
        metrics['combat_latency'] = combat_df['total_latency'].mean() if len(combat_df) > 0 else 0
        metrics['peaceful_latency'] = peaceful_df['total_latency'].mean() if len(peaceful_df) > 0 else 0
        
        # Save results
        df.to_csv(f'aws_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
        
        # Create visualizations
        self.create_visualizations(df, metrics)
        
        return metrics
    
    def create_visualizations(self, df, metrics):
        """Create performance visualizations"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Combat Prediction vs Reality
        ax = axes[0]
        ax.plot(df['timestamps'], df['combat_probability'], 'b-', label='LSTM Predictions', alpha=0.8)
        ax.fill_between(df['timestamps'], 0, df['actual_combat'], 
                       alpha=0.3, color='red', label='Actual Combat')
        ax.set_ylabel('Combat Probability')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Combat Predictions vs Reality (Accuracy: {metrics["accuracy"]:.1f}%)')
        
        # Plot 2: Network Slice Allocation
        ax = axes[1]
        ax.plot(df['timestamps'], df['slice_decision'], 'g-', linewidth=2)
        ax.fill_between(df['timestamps'], 0, df['actual_combat']*2, 
                       alpha=0.2, color='red')
        ax.set_ylabel('Network Slice')
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Basic', 'Medium', 'Premium'])
        ax.grid(True, alpha=0.3)
        ax.set_title('Q-Learning Network Slice Decisions')
        
        # Plot 3: Latency Performance
        ax = axes[2]
        ax.plot(df['timestamps'], df['total_latency'], 'purple', alpha=0.7)
        ax.axhline(y=50, color='green', linestyle='--', label='Target (<50ms)')
        ax.axhline(y=100, color='red', linestyle='--', label='Poor (>100ms)')
        ax.fill_between(df['timestamps'], 0, df['actual_combat']*150, 
                       alpha=0.2, color='red')
        ax.set_ylabel('Latency (ms)')
        ax.set_xlabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'System Latency (Avg: {metrics["avg_total_latency"]:.1f}ms)')
        
        plt.suptitle('AWS Deployment Performance - Edge + Cloud System', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'aws_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=150)
        plt.show()
    
    def print_summary(self, metrics):
        """Print formatted summary"""
        print("\n" + "="*60)
        print("🏆 AWS DEPLOYMENT PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"\n📊 Prediction Performance:")
        print(f"   Total Predictions: {metrics['total_predictions']}")
        print(f"   Correct Predictions: {metrics['correct_predictions']}")
        print(f"   Accuracy: {metrics['accuracy']:.1f}%")
        
        print(f"\n⚡ Latency Performance:")
        print(f"   Edge Server (LSTM): {metrics['avg_edge_latency']:.1f}ms")
        print(f"   Cloud Server (Q-Learning): {metrics['avg_cloud_latency']:.1f}ms")
        print(f"   Total Average: {metrics['avg_total_latency']:.1f}ms")
        print(f"   Under 50ms: {metrics['latency_under_50ms']:.1f}%")
        
        print(f"\n⚔️ Combat vs Peaceful:")
        print(f"   Combat Latency: {metrics['combat_latency']:.1f}ms")
        print(f"   Peaceful Latency: {metrics['peaceful_latency']:.1f}ms")
        
        print(f"\n📡 Network Slice Usage:")
        slice_names = {0: 'Basic', 1: 'Medium', 2: 'Premium'}
        total_slices = sum(metrics['slice_distribution'].values())
        for slice_id, count in sorted(metrics['slice_distribution'].items()):
            percentage = count / total_slices * 100
            print(f"   {slice_names[slice_id]}: {percentage:.1f}%")
        
        print("\n✅ Files saved:")
        print("   - aws_test_results_[timestamp].csv")
        print("   - aws_performance_[timestamp].png")

# Main execution
if __name__ == "__main__":
    print("🚀 AWS CLOUD GAMING SYSTEM TEST")
    print("Testing your deployed Edge-Cloud system on AWS EC2")
    
    tester = AWSSystemTest()
    
    # Run 5-minute simulation
    metrics = tester.run_gaming_simulation(duration_seconds=300)
    
    # Print summary
    tester.print_summary(metrics)