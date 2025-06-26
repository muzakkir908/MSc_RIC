import time
import random
from model_inference import CombatPredictor

def simulate_realtime_gaming():
    """Simulate real-time gaming with predictions"""
    predictor = CombatPredictor()
    
    print("🎮 REAL-TIME COMBAT PREDICTION DEMO")
    print("=" * 50)
    print("Simulating 60 seconds of gameplay...")
    print("Press Ctrl+C to stop\n")
    
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
                print(f"\n⚔️ COMBAT STARTING at {t/10:.1f}s!")
            elif combat_phase and t - phase_start > random.randint(50, 150):
                combat_phase = False
                print(f"\n✅ Combat ended at {t/10:.1f}s")
            
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
                print(f"\r[{t/10:4.1f}s] {status} | {pred_str} ({prob:.0%} confidence)", end="")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")

if __name__ == "__main__":
    simulate_realtime_gaming()
