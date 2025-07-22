import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import pandas as pd
from collections import deque

class LSTMModel(nn.Module):
    """Same LSTM architecture as training"""
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 16)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class CombatPredictor:
    """Easy-to-use combat prediction class"""
    
    def __init__(self, model_dir='trained_model'):
        """Load the trained model"""
        print("🔄 Loading trained model...")
        
        # Load model
        model_path = f'{model_dir}/lstm_model.pth'
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize model
        config = checkpoint['model_config']
        self.model = LSTMModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        with open(f'{model_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load config
        with open(f'{model_dir}/config.json', 'r') as f:
            self.config = json.load(f)
        
        self.feature_columns = self.config['feature_columns']
        self.sequence_length = self.config['sequence_length']
        
        # Buffer for real-time prediction
        self.feature_buffer = deque(maxlen=self.sequence_length)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Performance: {checkpoint['performance']}")
    
    def predict_single(self, game_state):
        """
        Make a single prediction
        
        Args:
            game_state: Dictionary with current game metrics
            
        Returns:
            (prediction, probability, confidence)
        """
        # Extract features in correct order
        features = []
        for col in self.feature_columns:
            features.append(game_state.get(col, 0))
        
        # Scale features
        features_scaled = self.scaler.transform([features])[0]
        
        # Add to buffer
        self.feature_buffer.append(features_scaled)
        
        # Need full sequence for prediction
        if len(self.feature_buffer) < self.sequence_length:
            return 0, 0.0, 0.0  # Not enough data yet
        
        # Create sequence tensor
        sequence = np.array(list(self.feature_buffer))
        sequence_tensor = torch.FloatTensor([sequence])
        
        # Make prediction
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            
            combat_prob = probabilities[1].item()
            prediction = 1 if combat_prob > 0.5 else 0
            confidence = abs(combat_prob - 0.5) * 2  # 0 to 1 scale
        
        return prediction, combat_prob, confidence
    
    def get_network_recommendation(self, combat_prob, confidence):
        """
        Recommend network slice based on prediction
        
        Returns:
            (slice_level, reason)
        """
        if combat_prob > 0.8 and confidence > 0.6:
            return 2, f"High combat probability ({combat_prob:.1%}), allocate premium resources"
        elif combat_prob > 0.5:
            return 1, f"Moderate combat probability ({combat_prob:.1%}), allocate medium resources"
        else:
            return 0, f"Low combat probability ({combat_prob:.1%}), maintain basic resources"

def test_model_inference():
    """Test the model with sample data"""
    print("\n🧪 TESTING MODEL INFERENCE")
    print("=" * 50)
    
    # Initialize predictor
    predictor = CombatPredictor()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Peaceful Exploration",
            "data": {
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
        },
        {
            "name": "Pre-Combat (High Alert)",
            "data": {
                "mouse_speed": 450,
                "turning_rate": 350,
                "movement_keys": 2,
                "is_shooting": 0,
                "activity_score": 0.6,
                "keys_pressed": 4,
                "ping_ms": 55,
                "cpu_percent": 55,
                "mouse_speed_ma": 420,
                "activity_ma": 0.55,
                "combat_likelihood": 0.6
            }
        },
        {
            "name": "Active Combat",
            "data": {
                "mouse_speed": 850,
                "turning_rate": 620,
                "movement_keys": 3,
                "is_shooting": 1,
                "activity_score": 0.9,
                "keys_pressed": 6,
                "ping_ms": 65,
                "cpu_percent": 75,
                "mouse_speed_ma": 820,
                "activity_ma": 0.88,
                "combat_likelihood": 0.95
            }
        }
    ]
    
    # Build up sequence buffer first
    print("\nBuilding sequence buffer...")
    for _ in range(30):
        predictor.predict_single(test_scenarios[0]["data"])
    
    # Test each scenario
    print("\nTesting different game scenarios:")
    print("-" * 50)
    
    for scenario in test_scenarios:
        # Run scenario multiple times to fill buffer
        for _ in range(5):
            predictor.predict_single(scenario["data"])
        
        # Get final prediction
        prediction, probability, confidence = predictor.predict_single(scenario["data"])
        slice_rec, reason = predictor.get_network_recommendation(probability, confidence)
        
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Prediction: {'COMBAT' if prediction == 1 else 'NON-COMBAT'}")
        print(f"   Combat Probability: {probability:.1%}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Network Slice: {['Basic', 'Medium', 'Premium'][slice_rec]}")
        print(f"   Reason: {reason}")

def create_realtime_demo():
    """Create a real-time demo script"""
    demo_code = '''import time
import random
from model_inference import CombatPredictor

def simulate_realtime_gaming():
    """Simulate real-time gaming with predictions"""
    predictor = CombatPredictor()
    
    print("🎮 REAL-TIME COMBAT PREDICTION DEMO")
    print("=" * 50)
    print("Simulating 60 seconds of gameplay...")
    print("Press Ctrl+C to stop\\n")
    
    # Initialize with peaceful data
    base_state = {
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
    
    # Build buffer
    for _ in range(30):
        predictor.predict_single(base_state)
    
    combat_phase = False
    phase_start = 0
    
    try:
        for t in range(600):  # 60 seconds at 10Hz
            # Simulate combat phases
            if not combat_phase and random.random() < 0.02:
                combat_phase = True
                phase_start = t
                print(f"\\n⚔️ COMBAT STARTING at {t/10:.1f}s!")
            elif combat_phase and t - phase_start > random.randint(50, 150):
                combat_phase = False
                print(f"\\n✅ Combat ended at {t/10:.1f}s")
            
            # Update game state
            if combat_phase:
                base_state["mouse_speed"] = random.uniform(600, 1000)
                base_state["turning_rate"] = random.uniform(400, 800)
                base_state["is_shooting"] = random.choice([0, 1, 1, 1])
                base_state["activity_score"] = random.uniform(0.7, 0.95)
            else:
                base_state["mouse_speed"] = random.uniform(50, 300)
                base_state["turning_rate"] = random.uniform(30, 200)
                base_state["is_shooting"] = 0
                base_state["activity_score"] = random.uniform(0.1, 0.4)
            
            # Update moving averages
            base_state["mouse_speed_ma"] = base_state["mouse_speed"] * 0.9
            base_state["activity_ma"] = base_state["activity_score"] * 0.9
            base_state["combat_likelihood"] = base_state["activity_score"]
            
            # Make prediction
            pred, prob, conf = predictor.predict_single(base_state)
            
            # Display every second
            if t % 10 == 0:
                status = "⚔️ COMBAT" if combat_phase else "🚶 Exploring"
                pred_str = "🎯 Combat predicted!" if pred == 1 else "✅ Peaceful"
                print(f"\\r[{t/10:4.1f}s] {status} | {pred_str} ({prob:.0%} confidence)", end="")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\\n\\nDemo stopped by user")

if __name__ == "__main__":
    simulate_realtime_gaming()
'''
    
    with open('realtime_demo.py', 'w', encoding='utf-8') as f:
        f.write(demo_code)
    
    print("\n✅ Created realtime_demo.py")

if __name__ == "__main__":
    # Test model inference
    test_model_inference()
    
    # Create demo script
    create_realtime_demo()
    
    print("\n\n📝 NEXT STEPS:")
    print("1. Run this script to test your model")
    print("2. Run 'python realtime_demo.py' for live demo")
    print("3. Integrate with Q-learning (next phase)")