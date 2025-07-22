#!/usr/bin/env python3
"""
Edge Server - Windows Compatible Version
Runs close to the player for fast predictions
"""

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import time
import os
from pathlib import Path

app = Flask(__name__)

# Global variables for model
model = None
scaler = None
config = None
prediction_buffer = []

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

def load_model():
    """Load the trained model - Windows compatible paths"""
    global model, scaler, config
    
    print("Loading LSTM model...")
    
    # Get the current directory
    current_dir = Path(__file__).parent
    base_dir = current_dir.parent  # Go up one level to deployment folder
    models_dir = base_dir / 'models'
    
    print(f"Looking for models in: {models_dir}")
    
    try:
        # Load model
        model_path = models_dir / 'lstm_model.pth'
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(str(model_path), map_location='cpu')
        
        # Initialize model
        model_config = checkpoint['model_config']
        model = LSTMModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load scaler
        scaler_path = models_dir / 'scaler.pkl'
        print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load config
        config_path = models_dir / 'config.json'
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in models directory:")
        if models_dir.exists():
            for file in models_dir.iterdir():
                print(f"  - {file.name}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make combat prediction"""
    global prediction_buffer
    
    try:
        # Get game state from request
        game_state = request.json
        
        # Extract features in correct order
        features = []
        for col in config['feature_columns']:
            features.append(game_state.get(col, 0))
        
        # Scale features
        features_scaled = scaler.transform([features])[0]
        
        # Add to buffer
        prediction_buffer.append(features_scaled)
        
        # Keep buffer size to sequence length
        if len(prediction_buffer) > config['sequence_length']:
            prediction_buffer.pop(0)
        
        # Need full sequence for prediction
        if len(prediction_buffer) < config['sequence_length']:
            return jsonify({
                'prediction': 0,
                'probability': 0.0,
                'confidence': 0.0,
                'message': f'Building buffer... {len(prediction_buffer)}/{config["sequence_length"]}'
            })
        
        # Create sequence tensor
        sequence = np.array(prediction_buffer)
        sequence_tensor = torch.FloatTensor([sequence])
        
        # Make prediction
        with torch.no_grad():
            output = model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            
            combat_prob = probabilities[1].item()
            prediction = 1 if combat_prob > 0.5 else 0
            confidence = abs(combat_prob - 0.5) * 2
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(combat_prob),
            'confidence': float(confidence),
            'latency_ms': 0  # Will be measured by client
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@app.route('/reset', methods=['POST'])
def reset_buffer():
    """Reset prediction buffer"""
    global prediction_buffer
    prediction_buffer = []
    return jsonify({'status': 'buffer reset'})

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run server
    print("Starting Edge Server on port 5000...")
    print("Access at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)