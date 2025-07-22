#!/usr/bin/env python3
"""
Cloud Server - Windows Compatible Version
Makes network slice decisions using Q-Learning
"""

from flask import Flask, request, jsonify
import numpy as np
import pickle
import json
from collections import defaultdict
import time
import os
from pathlib import Path

app = Flask(__name__)

# Global Q-learning agent
q_agent = None
last_state = None
last_action = None

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
    """Load trained Q-learning agent - Windows compatible"""
    global q_agent
    
    print("Loading Q-learning agent...")
    
    # Get the current directory
    current_dir = Path(__file__).parent
    base_dir = current_dir.parent  # Go up one level to deployment folder
    models_dir = base_dir / 'models'
    
    model_path = models_dir / 'trained_q_learning_model.pkl'
    print(f"Looking for Q-model at: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        q_agent = QLearningAgent()
        q_agent.q_table = defaultdict(lambda: np.zeros(3), model_data['q_table'])
        q_agent.state_bins = model_data['state_bins']
        
        print(f"✅ Q-agent loaded with {len(q_agent.q_table)} states")
        
    except FileNotFoundError:
        print(f"❌ Q-learning model not found at {model_path}")
        print("Using default agent with random policy")
        q_agent = QLearningAgent()
    except Exception as e:
        print(f"❌ Error loading Q-agent: {e}")
        print("Using default agent")
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
    """Decide network slice based on prediction and network state"""
    global last_state, last_action
    
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
        
        # Store for learning (in production, you'd log this)
        last_state = state
        last_action = action
        
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

@app.route('/feedback', methods=['POST'])
def receive_feedback():
    """Receive performance feedback (for future learning)"""
    try:
        feedback = request.json
        
        # In production, log this for offline learning
        result = {
            'status': 'feedback received',
            'performance': feedback.get('latency', 0),
            'timestamp': time.time()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    print("Starting Cloud Server on port 5001...")
    print("Access at: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)