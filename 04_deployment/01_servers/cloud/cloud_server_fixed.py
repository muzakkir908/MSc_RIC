#!/usr/bin/env python3
"""
Cloud Server - Q-Learning Decision Maker
This is the CORRECT version with /decide endpoint
"""

from flask import Flask, request, jsonify
import numpy as np
import pickle
from collections import defaultdict
import time
import os
from pathlib import Path

app = Flask(__name__)

# Global Q-learning agent
q_agent = None

class QLearningAgent:
    """Q-Learning agent for network slice selection"""
    
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(3))
        self.state_bins = [
            np.array([0.0, 0.3, 0.7, 1.0]),      # Combat probability
            np.array([0.0, 0.33, 0.67, 1.0]),    # Latency (normalized)
            np.array([0.0, 0.5, 1.0]),           # Network quality
            np.array([0.0, 0.5, 1.0]),           # CPU load
            np.array([0.0, 0.33, 0.67, 1.0])     # Time since combat
        ]
        self.epsilon = 0.0  # No exploration in production
        
    def discretize_state(self, state):
        """Convert continuous state to discrete"""
        discrete_state = []
        for i, value in enumerate(state):
            bin_index = np.digitize(value, self.state_bins[i]) - 1
            bin_index = np.clip(bin_index, 0, len(self.state_bins[i]) - 2)
            discrete_state.append(bin_index)
        return tuple(discrete_state)
    
    def get_action(self, state):
        """Get best action for state"""
        discrete_state = self.discretize_state(state)
        q_values = self.q_table[discrete_state]
        return int(np.argmax(q_values))

def load_q_agent():
    """Load trained Q-learning agent"""
    global q_agent
    
    print("Loading Q-learning agent...")
    
    # Try to find the model file
    possible_paths = [
    str(Path(__file__).resolve().parents[3] / '03_models' / 'qlearning' / 'trained_q_learning_model.pkl')
    ]

    


    
    model_loaded = False
    for path in possible_paths:
        print("Looking for model at:", path)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                
                q_agent = QLearningAgent()
                q_agent.q_table = defaultdict(lambda: np.zeros(3), model_data['q_table'])
                q_agent.state_bins = model_data['state_bins']
                
                print(f"✅ Q-agent loaded from {path} with {len(q_agent.q_table)} states")
                model_loaded = True
                break
            except Exception as e:
                print(f"Error loading from {path}: {e}")
    
    if not model_loaded:
        print("⚠️ No Q-learning model found, using default agent")
        q_agent = QLearningAgent()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'q_agent_loaded': q_agent is not None,
        'states_explored': len(q_agent.q_table) if q_agent else 0,
        'timestamp': time.time()
    })

@app.route('/decide', methods=['POST'])
def decide_slice():
    """CRITICAL ENDPOINT - Decide network slice based on prediction and network state"""
    global q_agent
    
    try:
        data = request.json
        
        # Extract inputs
        combat_prob = data.get('combat_probability', 0)
        current_latency = data.get('current_latency', 50)
        network_quality = data.get('network_quality', 0.8)
        cpu_load = data.get('cpu_load', 0.5)
        time_since_combat = data.get('time_since_combat', 100)
        
        # Create state vector
        state = np.array([
            combat_prob,
            min(current_latency / 150, 1),  # Normalize latency
            network_quality,
            cpu_load,
            min(time_since_combat / 300, 1)  # Normalize time
        ])
        
        # Get Q-learning decision
        action = q_agent.get_action(state)
        
        # Network slice properties
        slices = {
            0: {'name': 'Basic', 'bandwidth': 50, 'latency': 80, 'cost': 0.1},
            1: {'name': 'Medium', 'bandwidth': 100, 'latency': 50, 'cost': 0.3},
            2: {'name': 'Premium', 'bandwidth': 200, 'latency': 30, 'cost': 0.6}
        }
        
        selected_slice = slices[action]
        
        # Recommendation reasoning
        if combat_prob > 0.7:
            reason = f"High combat probability ({combat_prob:.0%})"
        elif current_latency > 80:
            reason = f"High latency ({current_latency:.0f}ms)"
        else:
            reason = "Normal conditions"
        
        return jsonify({
            'action': action,
            'slice': selected_slice['name'],
            'bandwidth_mbps': selected_slice['bandwidth'],
            'expected_latency': selected_slice['latency'],
            'cost_per_minute': selected_slice['cost'],
            'reason': reason,
            'state': state.tolist()
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    if not q_agent:
        return jsonify({'error': 'Q-agent not loaded'}), 500
    
    # Analyze Q-table policy
    policy_stats = {'basic': 0, 'medium': 0, 'premium': 0}
    
    for state, q_values in q_agent.q_table.items():
        best_action = np.argmax(q_values)
        if best_action == 0:
            policy_stats['basic'] += 1
        elif best_action == 1:
            policy_stats['medium'] += 1
        else:
            policy_stats['premium'] += 1
    
    total = sum(policy_stats.values())
    
    return jsonify({
        'total_states': len(q_agent.q_table),
        'policy_distribution': {
            k: f"{v/total*100:.1f}%" if total > 0 else "0%" 
            for k, v in policy_stats.items()
        }
    })

if __name__ == '__main__':
    # Load Q-agent on startup
    load_q_agent()
    
    # Run server
    print("Starting Cloud Server (Q-Learning) on port 5001...")
    print("Endpoints: /health, /decide, /stats")
    app.run(host='0.0.0.0', port=5001, debug=False)