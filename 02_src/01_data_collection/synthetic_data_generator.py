import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class SyntheticGameDataGenerator:
    def __init__(self, base_timestamp="2025-06-03T19:12:02.046805"):
        self.base_timestamp = pd.to_datetime(base_timestamp)
        self.scenarios = {
            'stable_home_wifi': {
                'probability': 0.3,
                'ping_base': 25, 'ping_std': 5,
                'packet_loss': 0.0,
                'bandwidth_recv': (100, 300),
                'bandwidth_sent': (20, 50)
            },
            'congested_network': {
                'probability': 0.2,
                'ping_base': 80, 'ping_std': 30,
                'packet_loss': 0.02,
                'bandwidth_recv': (50, 150),
                'bandwidth_sent': (10, 30)
            },
            'mobile': {
                'probability': 0.2,
                'ping_base': 60, 'ping_std': 20,
                'packet_loss': 0.01,
                'bandwidth_recv': (80, 200),
                'bandwidth_sent': (15, 40)
            },
            'unstable_connection': {
                'probability': 0.15,
                'ping_base': 100, 'ping_std': 50,
                'packet_loss': 0.05,
                'bandwidth_recv': (20, 100),
                'bandwidth_sent': (5, 20)
            },
            'peak_hours': {
                'probability': 0.15,
                'ping_base': 120, 'ping_std': 40,
                'packet_loss': 0.03,
                'bandwidth_recv': (30, 80),
                'bandwidth_sent': (8, 25)
            }
        }
        
    def generate_combat_pattern(self, n_rows):
        """Generate realistic combat patterns"""
        pattern = np.zeros(n_rows, dtype=bool)
        i = 0
        
        while i < n_rows:
            # Non-combat period (exploration/looting)
            non_combat_duration = np.random.randint(300, 1200)  # 30s to 2min
            pattern[i:i+non_combat_duration] = False
            i += non_combat_duration
            
            # Combat period
            if i < n_rows:
                combat_duration = np.random.randint(100, 600)  # 10s to 1min
                pattern[i:i+combat_duration] = True
                i += combat_duration
                
        return pattern[:n_rows]
    
    def generate_player_behavior(self, is_combat, scenario_name):
        """Generate realistic player behavior based on combat state"""
        if is_combat:
            # High activity during combat
            mouse_speed = np.random.normal(800, 300)
            turning_rate = np.random.normal(600, 200)
            movement_keys = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
            is_shooting = np.random.choice([True, False], p=[0.7, 0.3])
            keys_pressed = np.random.randint(3, 7)
        else:
            # Lower activity during non-combat
            mouse_speed = np.random.normal(200, 100)
            turning_rate = np.random.normal(150, 50)
            movement_keys = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            is_shooting = np.random.choice([True, False], p=[0.05, 0.95])
            keys_pressed = np.random.randint(0, 3)
            
        # Ensure non-negative values
        mouse_speed = max(0, mouse_speed)
        turning_rate = max(0, turning_rate)
        
        # Calculate activity score
        activity_score = (
            min(mouse_speed / 1000, 1) * 0.25 + 
            min(turning_rate / 500, 1) * 0.25 +
            movement_keys * 0.1 +
            is_shooting * 0.2 +
            np.random.uniform(0, 0.2)  # Random component
        )
        
        return {
            'mouse_speed': round(mouse_speed, 2),
            'turning_rate': round(turning_rate, 2),
            'movement_keys': movement_keys,
            'is_shooting': is_shooting,
            'keys_pressed': keys_pressed,
            'activity_score': round(activity_score, 3)
        }
    
    def generate_network_metrics(self, scenario, is_combat, previous_ping=None):
        """Generate network metrics based on scenario"""
        config = self.scenarios[scenario]
        
        # Base ping with some persistence (network conditions don't change instantly)
        if previous_ping and np.random.random() > 0.3:
            # 70% chance to stay near previous ping
            ping = np.random.normal(previous_ping, config['ping_std'] * 0.5)
        else:
            ping = np.random.normal(config['ping_base'], config['ping_std'])
        
        # Combat increases network load
        if is_combat:
            ping *= np.random.uniform(1.1, 1.5)
            bandwidth_multiplier = np.random.uniform(1.5, 3.0)
        else:
            bandwidth_multiplier = 1.0
            
        # Add occasional spikes
        if np.random.random() < 0.05:  # 5% chance of spike
            ping *= np.random.uniform(2, 4)
            
        # Ensure realistic bounds
        ping = np.clip(ping, 15, 500)
        
        # Bandwidth
        recv_kbs = np.random.uniform(*config['bandwidth_recv']) * bandwidth_multiplier
        sent_kbs = np.random.uniform(*config['bandwidth_sent']) * bandwidth_multiplier
        
        return {
            'ping_ms': round(ping, 1),
            'bytes_recv_kbs': round(recv_kbs, 2),
            'bytes_sent_kbs': round(sent_kbs, 2)
        }
    
    def generate_system_metrics(self, is_combat, scenario):
        """Generate system performance metrics"""
        if is_combat:
            # Higher resource usage during combat
            cpu_base = np.random.uniform(35, 60)
            gpu_base = np.random.uniform(70, 95)
        else:
            cpu_base = np.random.uniform(20, 35)
            gpu_base = np.random.uniform(50, 75)
            
        # Unstable scenarios might have performance issues
        if scenario in ['unstable_connection', 'peak_hours']:
            cpu_base *= np.random.uniform(1.1, 1.3)
            gpu_base *= np.random.uniform(1.05, 1.15)
            
        return {
            'cpu_percent': round(np.clip(cpu_base + np.random.normal(0, 5), 10, 100), 1),
            'cpu_freq_mhz': np.random.choice([1506, 2300], p=[0.3, 0.7]),
            'memory_percent': round(np.random.uniform(85, 92), 1),
            'gpu_percent': round(np.clip(gpu_base + np.random.normal(0, 5), 0, 100), 1),
            'gpu_memory_percent': round(np.random.uniform(60, 85), 1),
            'gpu_temp': round(np.random.uniform(65, 80), 1)
        }
    
    def generate_synthetic_data(self, n_rows=18000):
        """Generate complete synthetic dataset"""
        print(f"🎮 Generating {n_rows} rows of synthetic FreeFire data...")
        
        # Generate combat pattern
        combat_pattern = self.generate_combat_pattern(n_rows)
        
        # Initialize data storage
        data = []
        
        # Track previous values for continuity
        prev_ping = 30
        current_scenario = np.random.choice(list(self.scenarios.keys()))
        scenario_duration = 0
        
        # Mouse position for continuity
        mouse_x, mouse_y = 2230, 1762  # Starting from your last position
        
        for i in range(n_rows):
            # Change scenario occasionally
            scenario_duration += 1
            if scenario_duration > np.random.randint(300, 1500):  # 30s to 2.5min
                current_scenario = np.random.choice(
                    list(self.scenarios.keys()),
                    p=[s['probability'] for s in self.scenarios.values()]
                )
                scenario_duration = 0
                print(f"  Switching to scenario: {current_scenario} at row {i}")
            
            # Generate timestamp
            timestamp = self.base_timestamp + timedelta(milliseconds=100 * i)
            
            # Combat state
            is_combat = combat_pattern[i]
            
            # Player behavior
            behavior = self.generate_player_behavior(is_combat, current_scenario)
            
            # Update mouse position based on movement
            if behavior['mouse_speed'] > 100:
                mouse_x += np.random.randint(-50, 51)
                mouse_y += np.random.randint(-30, 31)
                mouse_x = np.clip(mouse_x, 0, 3840)  # Screen bounds
                mouse_y = np.clip(mouse_y, 0, 2160)
            
            # Network metrics
            network = self.generate_network_metrics(current_scenario, is_combat, prev_ping)
            prev_ping = network['ping_ms']
            
            # System metrics
            system = self.generate_system_metrics(is_combat, current_scenario)
            
            # Combine all data
            row = {
                'timestamp': timestamp.isoformat(),
                'mouse_x': mouse_x,
                'mouse_y': mouse_y,
                'mouse_speed': behavior['mouse_speed'],
                'turning_rate': behavior['turning_rate'],
                'is_shooting': behavior['is_shooting'],
                'movement_keys': behavior['movement_keys'],
                'keys_pressed': behavior['keys_pressed'],
                'is_combat': is_combat,
                'activity_score': behavior['activity_score'],
                'ping_ms': network['ping_ms'],
                'bytes_sent_kbs': network['bytes_sent_kbs'],
                'bytes_recv_kbs': network['bytes_recv_kbs'],
                'cpu_percent': system['cpu_percent'],
                'cpu_freq_mhz': system['cpu_freq_mhz'],
                'memory_percent': system['memory_percent'],
                'gpu_percent': system['gpu_percent'],
                'gpu_memory_percent': system['gpu_memory_percent'],
                'gpu_temp': system['gpu_temp']
            }
            
            data.append(row)
            
            # Progress update
            if (i + 1) % 3000 == 0:
                print(f"  Generated {i + 1}/{n_rows} rows...")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        filename = f"synthetic_game_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n✅ Synthetic data generated successfully!")
        print(f"📁 Saved to: {filename}")
        
        # Print scenario statistics
        self.print_scenario_stats(df)
        
        return df
    
    def print_scenario_stats(self, df):
        """Print statistics about generated scenarios"""
        print("\n📊 SYNTHETIC DATA STATISTICS:")
        print("=" * 50)
        
        # Combat statistics
        combat_ratio = df['is_combat'].mean() * 100
        print(f"\n⚔️ Combat: {combat_ratio:.1f}% of time")
        
        # Network statistics by scenario
        print("\n🌐 Network Conditions Generated:")
        print(f"  Ping range: {df['ping_ms'].min():.1f} - {df['ping_ms'].max():.1f} ms")
        print(f"  Average ping: {df['ping_ms'].mean():.1f} ms")
        print(f"  Ping > 100ms: {(df['ping_ms'] > 100).mean() * 100:.1f}% of time")
        print(f"  Ping > 150ms: {(df['ping_ms'] > 150).mean() * 100:.1f}% of time")
        
        # Performance issues
        print("\n⚠️ Performance Challenges:")
        high_ping_combat = df[df['is_combat'] & (df['ping_ms'] > 100)]
        print(f"  High ping during combat: {len(high_ping_combat)} events")
        print(f"  CPU > 80%: {(df['cpu_percent'] > 80).sum()} events")
        print(f"  GPU > 90%: {(df['gpu_percent'] > 90).sum()} events")
        
        print("\n💡 This synthetic data includes:")
        print("  ✓ Stable home WiFi scenarios")
        print("  ✓ Network congestion periods")
        print("  ✓ Mobile 4G connectivity")
        print("  ✓ Unstable connection drops")
        print("  ✓ Peak hour network stress")

if __name__ == "__main__":
    # Generate synthetic data
    generator = SyntheticGameDataGenerator()
    synthetic_df = generator.generate_synthetic_data(n_rows=18000)
    
    print("\n🔄 To merge with your real data:")
    print("real_df = pd.read_csv('your_real_data.csv')")
    print("synthetic_df = pd.read_csv('synthetic_game_data_[timestamp].csv')")
    print("merged_df = pd.concat([real_df, synthetic_df], ignore_index=True)")
    print("merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)")
    print("merged_df.to_csv('merged_enhanced_game_data.csv', index=False)")