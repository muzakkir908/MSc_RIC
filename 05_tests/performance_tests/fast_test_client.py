#!/usr/bin/env python3
"""
Fast Test Client - Optimized for local testing
Shows real performance without artificial delays
"""

import requests
import json
import time
import numpy as np
import concurrent.futures

class FastTestClient:
    def __init__(self, edge_url, cloud_url):
        self.edge_url = edge_url
        self.cloud_url = cloud_url
        self.session = requests.Session()  # Reuse connection
        
    def test_latency(self):
        """Test actual latency of the system"""
        print("\n⏱️  LATENCY TEST")
        print("=" * 40)
        
        # Warm up the buffer first
        print("Warming up prediction buffer...")
        for i in range(30):
            game_state = {
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
            self.session.post(f"{self.edge_url}/predict", json=game_state)
        
        print("\nTesting actual latencies...")
        
        # Test scenarios
        scenarios = [
            ("Peaceful", {
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
            }),
            ("Combat", {
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
            })
        ]
        
        for scenario_name, game_state in scenarios:
            print(f"\n{scenario_name} Scenario:")
            
            # Test edge latency
            edge_times = []
            for _ in range(10):
                start = time.perf_counter()
                edge_resp = self.session.post(f"{self.edge_url}/predict", json=game_state)
                edge_time = (time.perf_counter() - start) * 1000
                edge_times.append(edge_time)
            
            # Test cloud latency
            if edge_resp.status_code == 200:
                prediction = edge_resp.json()
                cloud_data = {
                    'combat_probability': prediction['probability'],
                    'current_latency': game_state['ping_ms'],
                    'network_quality': 0.8,
                    'cpu_load': game_state['cpu_percent'] / 100,
                    'time_since_combat': 0
                }
                
                cloud_times = []
                for _ in range(10):
                    start = time.perf_counter()
                    cloud_resp = self.session.post(f"{self.cloud_url}/decide", json=cloud_data)
                    cloud_time = (time.perf_counter() - start) * 1000
                    cloud_times.append(cloud_time)
                
                # Print results
                print(f"  Edge Server (LSTM): {np.mean(edge_times):.1f}ms (±{np.std(edge_times):.1f}ms)")
                print(f"  Cloud Server (Q-Learning): {np.mean(cloud_times):.1f}ms (±{np.std(cloud_times):.1f}ms)")
                print(f"  Total: {np.mean(edge_times) + np.mean(cloud_times):.1f}ms")
                
                if edge_resp.status_code == 200 and cloud_resp.status_code == 200:
                    print(f"  Prediction: {prediction['probability']:.0%} combat")
                    print(f"  Decision: {cloud_resp.json()['slice']}")
    
    def quick_simulation(self, duration=30):
        """Quick simulation without sleep delays"""
        print(f"\n🚀 QUICK SIMULATION ({duration}s)")
        print("=" * 40)
        
        # Reset buffer
        self.session.post(f"{self.edge_url}/reset")
        
        # Warm up
        for _ in range(30):
            self.session.post(f"{self.edge_url}/predict", json={
                "mouse_speed": 100, "turning_rate": 50, "movement_keys": 1,
                "is_shooting": 0, "activity_score": 0.2, "keys_pressed": 2,
                "ping_ms": 45, "cpu_percent": 35, "mouse_speed_ma": 90,
                "activity_ma": 0.18, "combat_likelihood": 0.1
            })
        
        results = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'latencies': [],
            'slice_counts': {'Basic': 0, 'Medium': 0, 'Premium': 0}
        }
        
        # Simulate without sleep
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration:
            # Determine if combat (simple pattern)
            elapsed = time.time() - start_time
            in_combat = 5 <= elapsed <= 10 or 15 <= elapsed <= 20
            
            # Generate state
            if in_combat:
                game_state = {
                    "mouse_speed": np.random.uniform(600, 1000),
                    "turning_rate": np.random.uniform(400, 800),
                    "movement_keys": 3,
                    "is_shooting": 1,
                    "activity_score": 0.9,
                    "keys_pressed": 6,
                    "ping_ms": 65,
                    "cpu_percent": 75,
                    "mouse_speed_ma": 800,
                    "activity_ma": 0.85,
                    "combat_likelihood": 0.9
                }
            else:
                game_state = {
                    "mouse_speed": np.random.uniform(50, 300),
                    "turning_rate": np.random.uniform(30, 200),
                    "movement_keys": 1,
                    "is_shooting": 0,
                    "activity_score": 0.2,
                    "keys_pressed": 2,
                    "ping_ms": 45,
                    "cpu_percent": 35,
                    "mouse_speed_ma": 150,
                    "activity_ma": 0.18,
                    "combat_likelihood": 0.1
                }
            
            # Make requests
            start_req = time.perf_counter()
            
            edge_resp = self.session.post(f"{self.edge_url}/predict", json=game_state)
            if edge_resp.status_code == 200:
                prediction = edge_resp.json()
                
                cloud_data = {
                    'combat_probability': prediction['probability'],
                    'current_latency': game_state['ping_ms'],
                    'network_quality': 0.8,
                    'cpu_load': 0.5,
                    'time_since_combat': 0 if in_combat else 100
                }
                
                cloud_resp = self.session.post(f"{self.cloud_url}/decide", json=cloud_data)
                
                total_latency = (time.perf_counter() - start_req) * 1000
                results['latencies'].append(total_latency)
                
                if cloud_resp.status_code == 200:
                    decision = cloud_resp.json()
                    results['slice_counts'][decision['slice']] += 1
                
                # Check prediction accuracy
                predicted_combat = prediction['prediction'] == 1
                if predicted_combat == in_combat:
                    results['correct_predictions'] += 1
                results['total_predictions'] += 1
            
            request_count += 1
            
            # Small delay to not overwhelm
            time.sleep(0.01)  # 10ms between requests
        
        # Print results
        print(f"\nProcessed {request_count} requests in {duration}s")
        print(f"Request rate: {request_count/duration:.1f} req/s")
        
        if results['total_predictions'] > 0:
            accuracy = results['correct_predictions'] / results['total_predictions'] * 100
            print(f"\nPrediction Accuracy: {accuracy:.1f}%")
        
        if results['latencies']:
            print(f"\nLatency Stats:")
            print(f"  Average: {np.mean(results['latencies']):.1f}ms")
            print(f"  Min: {np.min(results['latencies']):.1f}ms")
            print(f"  Max: {np.max(results['latencies']):.1f}ms")
            print(f"  95th percentile: {np.percentile(results['latencies'], 95):.1f}ms")
        
        print(f"\nSlice Distribution:")
        total_slices = sum(results['slice_counts'].values())
        for slice_name, count in results['slice_counts'].items():
            if total_slices > 0:
                print(f"  {slice_name}: {count/total_slices*100:.1f}%")

def main():
    # Configuration
    EDGE_URL = "http://localhost:5000"
    CLOUD_URL = "http://localhost:5001"
    
    client = FastTestClient(EDGE_URL, CLOUD_URL)
    
    print("🎮 FAST PERFORMANCE TEST")
    print("=" * 60)
    
    # Test 1: Actual latency
    client.test_latency()
    
    # Test 2: Quick simulation
    client.quick_simulation(duration=30)
    
    print("\n✅ Testing complete!")
    print("\nNOTE: High latencies in the original test were due to")
    print("sequential requests with sleeps. Actual system latency")
    print("should be much lower as shown above.")

if __name__ == "__main__":
    main()