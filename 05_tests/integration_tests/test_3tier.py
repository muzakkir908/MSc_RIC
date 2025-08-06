#!/usr/bin/env python3
"""
Test 3-Tier System: Edge -> Fog -> Cloud
"""

import requests
import time
import numpy as np
import matplotlib.pyplot as plt

class ThreeTierTest:
    def __init__(self, edge_ip, fog_ip, cloud_ip):
        self.edge_url = f"http://{edge_ip}:5000"
        self.fog_url = f"http://{fog_ip}:5002"
        self.cloud_url = f"http://{cloud_ip}:5001"



    def test_data_flow(self):
        """Test Edge -> Fog -> Cloud flow"""
        print("=" * 60)
        print("🔄 Testing 3-Tier Data Flow")
        print("=" * 60)

        # Warm up edge buffer
        print("1️⃣ Warming up Edge buffer...")
        for i in range(30):
            game_state = {
                "velocity": 0.5,
                "acceleration": 0.3,
                "orientation": 0.4,
                "action_freq": 0.2,
                "network_var": 0.1
            }
            requests.post(f"{self.edge_url}/predict", json=game_state)
        print("✅ Edge buffer ready")

        # Combat scenario
        print("\n2️⃣ Testing combat scenario...")
        combat_state = {
            "velocity": 0.9,
            "acceleration": 0.8,
            "orientation": 0.7,
            "action_freq": 0.95,
            "network_var": 0.4
        }

        # Edge prediction
        start = time.time()
        edge_resp = requests.post(f"{self.edge_url}/predict", json=combat_state)
        edge_time = (time.time() - start) * 1000
        edge_data = edge_resp.json()
        print(f"✅ Edge prediction: {edge_data['probability']:.2%} combat ({edge_time:.1f}ms)")

        # Fog processing
        fog_request = {
            'edge_prediction': edge_data,
            'current_latency': 65,
            'network_quality': 0.8,
            'cpu_load': 0.75,
            'time_since_combat': 0
        }

        start = time.time()
        fog_resp = requests.post(f"{self.fog_url}/process", json=fog_request)
        fog_time = (time.time() - start) * 1000
        fog_data = fog_resp.json()
        print(f"✅ Fog decision: {fog_data.get('slice', 'unknown')} slice via {fog_data.get('source', 'unknown')} ({fog_time:.1f}ms)")

        # Total time
        total_time = edge_time + fog_time
        print(f"\n📊 Total 3-tier latency: {total_time:.1f}ms")

        return edge_time, fog_time, total_time

    def test_caching_benefit(self):
        """Test fog caching performance"""
        print("\n🔄 Testing Fog Caching Benefits")
        print("=" * 60)

        # Clear cache first
        requests.post(f"{self.fog_url}/cache/clear")

        latencies_no_cache = []
        latencies_with_cache = []

        # Test without cache (first requests)
        print("Testing without cache...")
        fog_request = {
            'edge_prediction': {'probability': 0.8, 'prediction': 1},
            'current_latency': 65,
            'network_quality': 0.8,
            'cpu_load': 0.75,
            'time_since_combat': 0
        }

        for i in range(10):
            start = time.time()
            fog_resp = requests.post(f"{self.fog_url}/process", json=fog_request)
            latency = (time.time() - start) * 1000
            latencies_no_cache.append(latency)

        # Test with cache (repeated requests)
        print("Testing with cache...")
        for i in range(10):
            start = time.time()
            fog_resp = requests.post(f"{self.fog_url}/process", json=fog_request)
            latency = (time.time() - start) * 1000
            latencies_with_cache.append(latency)

        # Results
        avg_no_cache = np.mean(latencies_no_cache)
        avg_with_cache = np.mean(latencies_with_cache)
        improvement = (avg_no_cache - avg_with_cache) / avg_no_cache * 100

        print(f"\n📊 Caching Results:")
        print(f"   Without cache: {avg_no_cache:.1f}ms")
        print(f"   With cache: {avg_with_cache:.1f}ms")
        print(f"   Improvement: {improvement:.1f}%")

        return latencies_no_cache, latencies_with_cache

    def run_3tier_simulation(self, duration=120):
        """Run complete 3-tier simulation"""
        print(f"\n🎮 RUNNING 3-TIER SIMULATION ({duration}s)")
        print("=" * 60)

        results = {
            'timestamps': [],
            'edge_latency': [],
            'fog_latency': [],
            'total_latency': [],
            'slice_decision': [],
            'source': []
        }

        combat_phases = [(20, 40), (70, 90)]

        for t in range(duration):
            in_combat = any(start <= t < end for start, end in combat_phases)

            if in_combat:
                game_state = {
                    "velocity": np.random.uniform(0.7, 1.0),
                    "acceleration": np.random.uniform(0.7, 1.0),
                    "orientation": np.random.uniform(0.7, 1.0),
                    "action_freq": np.random.uniform(0.8, 1.0),
                    "network_var": np.random.uniform(0.3, 0.5)
                }
            else:
                game_state = {
                    "velocity": np.random.uniform(0.1, 0.4),
                    "acceleration": np.random.uniform(0.1, 0.4),
                    "orientation": np.random.uniform(0.1, 0.4),
                    "action_freq": np.random.uniform(0.1, 0.3),
                    "network_var": np.random.uniform(0.1, 0.3)
                }

            # Edge
            start_edge = time.time()
            edge_resp = requests.post(f"{self.edge_url}/predict", json=game_state)
            edge_latency = (time.time() - start_edge) * 1000

            if edge_resp.status_code == 200:
                edge_data = edge_resp.json()

                # Fog
                fog_request = {
                    'edge_prediction': edge_data,
                    'current_latency': 65,
                    'network_quality': 0.8,
                    'cpu_load': 0.7,
                    'time_since_combat': 0 if in_combat else t
                }

                start_fog = time.time()
                fog_resp = requests.post(f"{self.fog_url}/process", json=fog_request)
                fog_latency = (time.time() - start_fog) * 1000

                if fog_resp.status_code == 200:
                    fog_data = fog_resp.json()

                    results['timestamps'].append(t)
                    results['edge_latency'].append(edge_latency)
                    results['fog_latency'].append(fog_latency)
                    results['total_latency'].append(edge_latency + fog_latency)
                    results['slice_decision'].append(fog_data.get('action', 0))
                    results['source'].append(fog_data.get('source', 'unknown'))

                    if t % 10 == 0:
                        status = "⚔️ COMBAT" if in_combat else "🚶 Peaceful"
                        print(f"[{t:3d}s] {status} | Edge: {edge_latency:.1f}ms | Fog: {fog_latency:.1f}ms | Source: {fog_data.get('source', 'unknown')}")

            time.sleep(0.1)

        self.visualize_3tier_results(results)

        return results

    def visualize_3tier_results(self, results):
        """Create 3-tier performance visualization"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: Latency breakdown
        ax = axes[0]
        ax.plot(results['timestamps'], results['edge_latency'], 'b-', label='Edge', alpha=0.7)
        ax.plot(results['timestamps'], results['fog_latency'], 'g-', label='Fog', alpha=0.7)
        ax.plot(results['timestamps'], results['total_latency'], 'r-', label='Total', linewidth=2)
        ax.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Target')
        ax.set_ylabel('Latency (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('3-Tier System Latency Performance')

        # Plot 2: Cache hits
        ax = axes[1]
        cache_hits = [1 if s == 'fog_cache' else 0 for s in results['source']]
        ax.plot(results['timestamps'], cache_hits, 'o-', color='green', alpha=0.6)
        ax.set_ylabel('Cache Hit')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.set_title('Fog Cache Utilization')

        # Plot 3: Slice decisions
        ax = axes[2]
        ax.plot(results['timestamps'], results['slice_decision'], 'purple', linewidth=2)
        ax.set_ylabel('Network Slice')
        ax.set_xlabel('Time (seconds)')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Basic', 'Medium', 'Premium'])
        ax.grid(True, alpha=0.3)
        ax.set_title('Network Slice Allocation')

        plt.suptitle('3-Tier Architecture Performance (Edge-Fog-Cloud)', fontsize=16)
        plt.tight_layout()
        plt.savefig('3tier_performance.png', dpi=150)
        plt.show()


if __name__ == "__main__":
    EDGE_IP = "54.146.235.60"     # ✅ EdgeServer (t2.medium)
    FOG_IP = "54.234.97.2"        # ✅ FogServer (t2.medium)
    CLOUD_IP = "98.81.90.202"     # ✅ CloudServer (t2.medium)


    tester = ThreeTierTest(EDGE_IP, FOG_IP, CLOUD_IP)

    # Test 1: Basic data flow
    edge_time, fog_time, total_time = tester.test_data_flow()

    # Test 2: Caching benefit
    no_cache, with_cache = tester.test_caching_benefit()

    # Test 3: Full simulation
    results = tester.run_3tier_simulation(duration=120)

    print("\n✅ 3-Tier testing complete!")

