#!/usr/bin/env python3
"""
Test Client - Simulates game data and tests the deployed system
"""

import requests
import json
import time
import numpy as np

class TestClient:
    def __init__(self, edge_url, cloud_url):
        self.edge_url = edge_url
        self.cloud_url = cloud_url
        self.game_state_buffer = []
        
    def test_health(self):
        """Test if servers are running"""
        print("🔍 Testing server health...")
        
        # Test edge server
        try:
            resp = requests.get(f"{self.edge_url}/health", timeout=5)
            print(f"✅ Edge server: {resp.json()}")
        except Exception as e:
            print(f"❌ Edge server error: {e}")
            
        # Test cloud server
        try:
            resp = requests.get(f"{self.cloud_url}/health", timeout=5)
            print(f"✅ Cloud server: {resp.json()}")
        except Exception as e:
            print(f"❌ Cloud server error: {e}")
    
    def simulate_game_session(self, duration_seconds=60):
        """Simulate a game session"""
        print(f"\n🎮 Simulating {duration_seconds}s game session...")
        
        # Reset edge buffer
        requests.post(f"{self.edge_url}/reset")
        
        # Game phases
        combat_phases = [
            (10, 20),   # Combat at 10-20s
            (35, 45),   # Combat at 35-45s
        ]
        
        results = []
        
        for t in range(duration_seconds * 10):  # 10Hz
            time_sec = t / 10
            
            # Check if in combat
            in_combat = any(start <= time_sec < end for start, end in combat_phases)
            
            # Generate game state
            if in_combat:
                game_state = {
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
                game_state = {
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
            
            # Get prediction from edge
            start_time = time.time()
            edge_resp = requests.post(f"{self.edge_url}/predict", json=game_state)
            edge_latency = (time.time() - start_time) * 1000
            
            if edge_resp.status_code == 200:
                prediction = edge_resp.json()
                
                # Get network decision from cloud
                cloud_data = {
                    'combat_probability': prediction['probability'],
                    'current_latency': game_state['ping_ms'],
                    'network_quality': 0.8,
                    'cpu_load': game_state['cpu_percent'] / 100,
                    'time_since_combat': 0 if in_combat else t
                }
                
                start_time = time.time()
                cloud_resp = requests.post(f"{self.cloud_url}/decide", json=cloud_data)
                cloud_latency = (time.time() - start_time) * 1000
                
                if cloud_resp.status_code == 200:
                    decision = cloud_resp.json()
                    
                    result = {
                        'time': time_sec,
                        'actual_combat': in_combat,
                        'predicted_combat': prediction['prediction'],
                        'combat_probability': prediction['probability'],
                        'slice_decision': decision['slice'],
                        'edge_latency_ms': edge_latency,
                        'cloud_latency_ms': cloud_latency,
                        'total_latency_ms': edge_latency + cloud_latency
                    }
                    
                    results.append(result)
                    
                    # Print progress every second
                    if t % 10 == 0:
                        status = "⚔️ COMBAT" if in_combat else "🚶 Peaceful"
                        pred_status = "🎯 Combat predicted" if prediction['prediction'] else "✅ Peaceful"
                        print(f"[{time_sec:4.1f}s] {status} | {pred_status} ({prediction['probability']:.0%}) | Slice: {decision['slice']} | Latency: {edge_latency + cloud_latency:.1f}ms")
            
            time.sleep(0.1)  # 10Hz
        
        # Summary
        self.print_summary(results)
        return results
    
    def print_summary(self, results):
        """Print session summary"""
        print("\n📊 SESSION SUMMARY")
        print("=" * 60)
        
        if not results:
            print("No results to analyze")
            return
        
        # Accuracy
        correct = sum(1 for r in results if r['actual_combat'] == r['predicted_combat'])
        accuracy = correct / len(results) * 100
        print(f"Prediction Accuracy: {accuracy:.1f}%")
        
        # Latency
        avg_edge = np.mean([r['edge_latency_ms'] for r in results])
        avg_cloud = np.mean([r['cloud_latency_ms'] for r in results])
        avg_total = np.mean([r['total_latency_ms'] for r in results])
        
        print(f"\nLatency Performance:")
        print(f"  Edge (prediction): {avg_edge:.1f}ms")
        print(f"  Cloud (decision): {avg_cloud:.1f}ms")
        print(f"  Total: {avg_total:.1f}ms")
        
        # Slice usage
        slice_counts = {}
        for r in results:
            slice = r['slice_decision']
            slice_counts[slice] = slice_counts.get(slice, 0) + 1
        
        print(f"\nSlice Usage:")
        for slice, count in slice_counts.items():
            print(f"  {slice}: {count/len(results)*100:.1f}%")
    
    def test_single_prediction(self):
        """Test a single prediction"""
        print("\n🧪 Testing single prediction...")
        
        # Combat scenario
        combat_state = {
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
        
        # Get prediction
        resp = requests.post(f"{self.edge_url}/predict", json=combat_state)
        if resp.status_code == 200:
            prediction = resp.json()
            print(f"✅ Combat prediction: {prediction}")
        else:
            print(f"❌ Prediction failed: {resp.text}")

def main():
    """Run tests"""
    # Configuration
    EDGE_URL = "http://localhost:5000"  # Change to EC2 URL later
    CLOUD_URL = "http://localhost:5001"  # Change to EC2 URL later
    
    client = TestClient(EDGE_URL, CLOUD_URL)
    
    print("🎮 CLOUD GAMING DEPLOYMENT TEST")
    print("=" * 60)
    
    # Test 1: Health check
    client.test_health()
    
    # Test 2: Single prediction
    client.test_single_prediction()
    
    # Test 3: Full simulation
    input("\nPress Enter to start full simulation...")
    client.simulate_game_session(duration_seconds=60)

if __name__ == "__main__":
    main()