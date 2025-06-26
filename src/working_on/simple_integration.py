import numpy as np
from model_inference import CombatPredictor
import matplotlib.pyplot as plt

class SimpleNetworkManager:
    """Simple decision maker using LSTM predictions"""
    
    def __init__(self):
        self.predictor = CombatPredictor()
        self.current_slice = 0  # Start with basic
        self.history = []
        
    def decide_network_slice(self, game_state, network_state):
        """
        Simple rule-based decision using LSTM prediction
        
        This is what Q-learning will replace with learned decisions
        """
        # Get combat prediction
        pred, prob, conf = self.predictor.predict_single(game_state)
        
        # Simple rules (Q-learning will learn these automatically)
        if prob > 0.7:  # High combat probability
            recommended_slice = 2  # Premium
        elif prob > 0.3:  # Medium probability
            recommended_slice = 1  # Medium
        else:
            recommended_slice = 0  # Basic
        
        # Consider network conditions
        if network_state['latency'] > 80:  # High latency
            recommended_slice = min(recommended_slice + 1, 2)
        
        # Record decision
        self.history.append({
            'combat_prob': prob,
            'latency': network_state['latency'],
            'slice': recommended_slice,
            'actual_combat': game_state.get('is_combat', 0)
        })
        
        self.current_slice = recommended_slice
        return recommended_slice
    
    def calculate_performance(self):
        """Calculate how well we did"""
        if not self.history:
            return
        
        correct_predictions = 0
        resource_waste = 0
        latency_violations = 0
        
        for record in self.history:
            # Good: Premium slice when combat
            if record['actual_combat'] and record['slice'] == 2:
                correct_predictions += 1
            # Bad: Basic slice during combat
            elif record['actual_combat'] and record['slice'] == 0:
                latency_violations += 1
            # Waste: Premium when not needed
            elif not record['actual_combat'] and record['slice'] == 2:
                resource_waste += 1
        
        total = len(self.history)
        print(f"\n📊 Performance Summary:")
        print(f"   Correct allocations: {correct_predictions}/{total} ({correct_predictions/total*100:.1f}%)")
        print(f"   Resource waste: {resource_waste}/{total} ({resource_waste/total*100:.1f}%)")
        print(f"   Latency violations: {latency_violations}/{total} ({latency_violations/total*100:.1f}%)")

def simulate_game_session():
    """Simulate a gaming session with network management"""
    print("🎮 SIMULATING GAME SESSION WITH NETWORK MANAGEMENT")
    print("=" * 60)
    
    manager = SimpleNetworkManager()
    
    # Simulate 5 minutes of gameplay
    duration = 300  # seconds
    timesteps = duration * 10  # 10Hz
    
    # Game phases
    game_states = []
    network_states = []
    
    print("\nGenerating game scenario...")
    
    # Create realistic game phases
    combat_phases = [
        (500, 700),    # Combat at 50-70s
        (1200, 1500),  # Combat at 120-150s
        (2000, 2300),  # Combat at 200-230s
        (2600, 2800),  # Combat at 260-280s
    ]
    
    for t in range(timesteps):
        # Check if in combat phase
        in_combat = any(start <= t < end for start, end in combat_phases)
        
        # Generate game state
        if in_combat:
            game_state = {
                "mouse_speed": np.random.uniform(600, 1000),
                "turning_rate": np.random.uniform(400, 800),
                "movement_keys": np.random.randint(2, 4),
                "is_shooting": np.random.choice([0, 1], p=[0.3, 0.7]),
                "activity_score": np.random.uniform(0.7, 0.95),
                "keys_pressed": np.random.randint(4, 7),
                "ping_ms": np.random.uniform(50, 80),
                "cpu_percent": np.random.uniform(60, 85),
                "mouse_speed_ma": 0,  # Will be calculated
                "activity_ma": 0,      # Will be calculated
                "combat_likelihood": 0, # Will be calculated
                "is_combat": 1
            }
        else:
            game_state = {
                "mouse_speed": np.random.uniform(50, 300),
                "turning_rate": np.random.uniform(30, 200),
                "movement_keys": np.random.randint(0, 2),
                "is_shooting": 0,
                "activity_score": np.random.uniform(0.1, 0.4),
                "keys_pressed": np.random.randint(1, 3),
                "ping_ms": np.random.uniform(35, 60),
                "cpu_percent": np.random.uniform(30, 50),
                "mouse_speed_ma": 0,
                "activity_ma": 0,
                "combat_likelihood": 0,
                "is_combat": 0
            }
        
        # Calculate moving averages
        game_state["mouse_speed_ma"] = game_state["mouse_speed"] * 0.9
        game_state["activity_ma"] = game_state["activity_score"] * 0.9
        game_state["combat_likelihood"] = game_state["activity_score"]
        
        # Network state (affected by slice choice)
        base_latency = np.random.uniform(40, 60)
        if hasattr(manager, 'current_slice'):
            if manager.current_slice == 0:  # Basic
                latency = base_latency * np.random.uniform(1.5, 2.0)
            elif manager.current_slice == 1:  # Medium
                latency = base_latency * np.random.uniform(1.0, 1.3)
            else:  # Premium
                latency = base_latency * np.random.uniform(0.6, 0.8)
        else:
            latency = base_latency
        
        network_state = {
            "latency": latency,
            "bandwidth": np.random.uniform(50, 200),
            "packet_loss": np.random.uniform(0, 0.02)
        }
        
        game_states.append(game_state)
        network_states.append(network_state)
    
    print("Running network management simulation...")
    
    # Build up buffer first
    for i in range(30):
        manager.decide_network_slice(game_states[i], network_states[i])
    
    # Run simulation
    results = []
    for i in range(30, timesteps):
        slice_choice = manager.decide_network_slice(game_states[i], network_states[i])
        
        if i % 100 == 0:  # Every 10 seconds
            combat_status = "⚔️ COMBAT" if game_states[i]['is_combat'] else "🚶 Peaceful"
            slice_name = ['Basic', 'Medium', 'Premium'][slice_choice]
            print(f"[{i/10:5.1f}s] {combat_status} | Network: {slice_name} | Latency: {network_states[i]['latency']:.1f}ms")
    
    # Show performance
    manager.calculate_performance()
    
    # Create visualization
    print("\nGenerating performance visualization...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Extract data for plotting
    times = np.arange(len(manager.history)) / 10
    combat_probs = [h['combat_prob'] for h in manager.history]
    slices = [h['slice'] for h in manager.history]
    latencies = [h['latency'] for h in manager.history]
    actual_combat = np.array([h['actual_combat'] for h in manager.history])
    
    # Plot 1: Combat probability and actual combat
    ax1.plot(times, combat_probs, 'b-', label='Predicted Combat Prob', alpha=0.7)
    ax1.fill_between(times, 0, actual_combat, alpha=0.3, color='red', label='Actual Combat')
    ax1.set_ylabel('Combat Probability')
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Combat Prediction vs Reality')
    
    # Plot 2: Network slice allocation
    ax2.plot(times, slices, 'g-', linewidth=2)
    ax2.fill_between(times, 0, actual_combat*2, alpha=0.2, color='red')
    ax2.set_ylabel('Network Slice')
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Basic', 'Medium', 'Premium'])
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Network Resource Allocation')
    
    # Plot 3: Latency
    ax3.plot(times, latencies, 'purple', alpha=0.7)
    ax3.axhline(y=50, color='green', linestyle='--', label='Target (<50ms)')
    ax3.axhline(y=100, color='red', linestyle='--', label='Poor (>100ms)')
    ax3.fill_between(times, 0, actual_combat*150, alpha=0.2, color='red')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_xlabel('Time (seconds)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Network Latency Performance')
    
    plt.tight_layout()
    plt.savefig('network_management_demo.png', dpi=150)
    plt.show()
    
    print("\n✅ Simulation complete! Check 'network_management_demo.png'")

if __name__ == "__main__":
    simulate_game_session()