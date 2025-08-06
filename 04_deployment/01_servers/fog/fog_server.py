#!/usr/bin/env python3
"""
Fog Server - Intermediate layer between Edge and Cloud
Provides caching and aggregation
"""

from flask import Flask, request, jsonify
import requests
import time
import json
from collections import deque
import numpy as np

app = Flask(__name__)

# Configuration
CLOUD_URL = "http://98.81.90.202:5001"

# Caching
prediction_cache = {}
decision_cache = {}
cache_ttl = 5  # seconds

# Aggregation buffer
prediction_buffer = deque(maxlen=10)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'role': 'fog',
        'cache_size': len(prediction_cache),
        'timestamp': time.time()
    })

@app.route('/process', methods=['POST'])
def process_prediction():
    """Fog processing: caching and aggregation"""
    try:
        data = request.get_json(force=True)
        print("📥 Received from Edge:\n", json.dumps(data, indent=2))  # ✅ Debug print
        
        # Extract edge prediction
        edge_prediction = data.get('edge_prediction', {})
        combat_prob = edge_prediction.get('probability', 0)
        
        # Add to aggregation buffer
        prediction_buffer.append(combat_prob)
        
        # Calculate aggregated metrics
        avg_combat_prob = np.mean(prediction_buffer)
        trend = "increasing" if len(prediction_buffer) > 1 and prediction_buffer[-1] > prediction_buffer[0] else "stable"
        
        # Check cache for similar states
        cache_key = f"{combat_prob:.2f}_{data.get('current_latency', 50)}"
        
        if cache_key in decision_cache:
            cached_decision = decision_cache[cache_key]
            if time.time() - cached_decision['timestamp'] < cache_ttl:
                # Return cached decision
                return jsonify({
                    'source': 'fog_cache',
                    'decision': cached_decision['decision'],
                    'aggregated_combat_prob': avg_combat_prob,
                    'trend': trend
                })
        
        # Forward to cloud with aggregated data
        cloud_request = {
            'combat_probability': combat_prob,
            'current_latency': data.get('current_latency', 50),
            'network_quality': data.get('network_quality', 0.8),
            'cpu_load': data.get('cpu_load', 0.5),
            'time_since_combat': data.get('time_since_combat', 100),
            'fog_aggregated_prob': avg_combat_prob,
            'fog_trend': trend
        }
        
        print("⛅ Forwarding to Cloud:", json.dumps(cloud_request, indent=2))  # ✅ Cloud request debug

        start_time = time.time()
        cloud_response = requests.post(f"{CLOUD_URL}/decide", json=cloud_request)
        duration = time.time() - start_time

        if cloud_response.status_code == 200:
            decision = cloud_response.json()
            
            # Cache the decision
            decision_cache[cache_key] = {
                'decision': decision,
                'timestamp': time.time()
            }
            
            # Add fog metadata
            decision['source'] = 'cloud_via_fog'
            decision['fog_aggregated_prob'] = avg_combat_prob
            decision['fog_trend'] = trend
            
            return jsonify(decision)
        else:
            print("❌ Cloud server error:", cloud_response.text)
            return jsonify({'error': 'Cloud server error'}), 500
            
    except Exception as e:
        print("🚨 Error in /process:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear fog cache"""
    prediction_cache.clear()
    decision_cache.clear()
    prediction_buffer.clear()
    return jsonify({'status': 'cache cleared'})

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get fog statistics"""
    return jsonify({
        'cache_entries': len(decision_cache),
        'buffer_size': len(prediction_buffer),
        'avg_combat_prob': np.mean(prediction_buffer) if prediction_buffer else 0,
        'cache_hit_potential': len(decision_cache) / (len(decision_cache) + 1) * 100
    })

if __name__ == '__main__':
    print("Starting Fog Server on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=False)
