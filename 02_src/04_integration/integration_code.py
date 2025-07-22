import numpy as np
import torch
import pandas as pd
import json
import requests
from collections import deque
from typing import Dict, Tuple, List
import time

class PredictionIntegration:
    """Integration layer between LSTM predictor and Q-learning agent"""
    
    def __init__(self, model_path='model_output/combat_prediction_lstm.pth', 
                 api_url=None, use_api=False):
        """
        Initialize integration layer
        
        Args:
            model_path: Path to local model file
            api_url: URL of deployed model API (e.g., http://edge-ip:8000)
            use_api: Whether to use API or local model
        """
        self.use_api = use_api
        self.api_url = api_url
        
        if not use_api:
            # Load local model
            from inference import CombatPredictor
            self.predictor = CombatPredictor(model_path)
        
        # Prediction cache for RL agent
        self.prediction_cache = deque(maxlen=100)
        self.confidence_threshold = 0.7
        
    def get_prediction(self, game_state: Dict) -> Tuple[float, int, float]:
        """
        Get combat prediction from model
        
        Args:
            game_state: Dictionary with current game features
            
        Returns:
            Tuple of (combat_probability, prediction, confidence)
        """
        if self.use_api:
            return self._get_api_prediction(game_state)
        else:
            return self._get_local_prediction(game_state)
    
    def _get_local_prediction(self, game_state: Dict) -> Tuple[float, int, float]:
        """Get prediction from local model"""
        # Extract features needed for model
        features = {
            'mouse_speed': game_state.get('mouse_speed', 0),
            'turning_rate': game_state.get('turning_rate', 0),
            'movement_keys': game_state.get('movement_keys', 0),
            'is_shooting': game_state.get('is_shooting', 0),
            'activity_score': game_state.get('activity_score', 0),
            'mouse_speed_ma': game_state.get('mouse_speed_ma', 0),
            'activity_ma': game_state.get('activity_ma', 0),
            'combat_likelihood': game_state.get('combat_likelihood', 0)
        }
        
        # Get prediction
        prob, pred = self.predictor.predict(features)
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(prob - 0.5) * 2
        
        # Cache result
        self.prediction_cache.append({
            'timestamp': time.time(),
            'probability': prob,
            'prediction': pred,
            'confidence': confidence
        })
        
        return prob, pred, confidence
    
    def _get_api_prediction(self, game_state: Dict) -> Tuple[float, int, float]:
        """Get prediction from deployed API"""
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json={'features': game_state, 'timestamp': time.time()}
            )
            
            if response.status_code == 200:
                data = response.json()
                prob = data['combat_probability']
                pred = 1 if data['prediction'] == 'combat' else 0
                confidence = abs(prob - 0.5) * 2
                
                # Cache result
                self.prediction_cache.append({
                    'timestamp': time.time(),
                    'probability': prob,
                    'prediction': pred,
                    'confidence': confidence
                })
                
                return prob, pred, confidence
            else:
                # Fallback to last known prediction
                if self.prediction_cache:
                    last = self.prediction_cache[-1]
                    return last['probability'], last['prediction'], last['confidence']
                return 0.0, 0, 0.0
                
        except Exception as e:
            print(f"API prediction error: {e}")
            return 0.0, 0, 0.0
    
    def get_rl_state(self, game_state: Dict, network_state: Dict) -> np.ndarray:
        """
        Convert game and network state to RL state vector
        
        Args:
            game_state: Current game metrics
            network_state: Current network metrics
            
        Returns:
            Normalized state vector for RL agent
        """
        # Get prediction
        combat_prob, _, confidence = self.get_prediction(game_state)
        
        # Create state vector
        state = np.array([
            # Combat prediction info
            combat_prob,                                    # 0-1
            confidence,                                     # 0-1
            
            # Network metrics
            np.clip(network_state['ping_ms'] / 200, 0, 1), # Normalized ping
            network_state.get('network_quality', 0.5),     # 0-1
            
            # System metrics
            network_state.get('system_stress', 0.5),       # 0-1
            
            # Player activity
            np.clip(game_state.get('activity_score', 0), 0, 1),
            
            # Recent performance
            network_state.get('performance_risk', 0),      # 0-1
            
            # Trend indicators
            self._calculate_prediction_trend()              # -1 to 1
        ])
        
        return state
    
    def _calculate_prediction_trend(self) -> float:
        """Calculate trend in combat predictions"""
        if len(self.prediction_cache) < 3:
            return 0.0
        
        # Get recent predictions
        recent = list(self.prediction_cache)[-10:]
        probs = [p['probability'] for p in recent]
        
        # Calculate trend (positive = increasing combat likelihood)
        if len(probs) >= 2:
            trend = np.mean(np.diff(probs))
            return np.clip(trend * 10, -1, 1)  # Scale and clip
        
        return 0.0
    
    def should_proactive_allocate(self, combat_prob: float, confidence: float, 
                                 current_slice: int) -> bool:
        """
        Determine if proactive allocation is needed
        
        Args:
            combat_prob: Predicted combat probability
            confidence: Prediction confidence
            current_slice: Current network slice (0=low, 1=med, 2=high)
            
        Returns:
            Boolean indicating if upgrade is recommended
        """
        # High confidence combat prediction
        if confidence > self.confidence_threshold and combat_prob > 0.7:
            return current_slice < 2  # Upgrade to high if not already
        
        # Medium confidence
        elif confidence > 0.5 and combat_prob > 0.5:
            return current_slice < 1  # At least medium slice
        
        return False
    
    def get_allocation_recommendation(self, rl_state: np.ndarray, 
                                    prediction_output: Tuple) -> Dict:
        """
        Get comprehensive allocation recommendation
        
        Args:
            rl_state: Current RL state vector
            prediction_output: (probability, prediction, confidence)
            
        Returns:
            Dictionary with allocation recommendation
        """
        combat_prob, prediction, confidence = prediction_output
        
        # Determine urgency
        urgency = 'high' if combat_prob > 0.8 else 'medium' if combat_prob > 0.5 else 'low'
        
        # Recommended action
        if combat_prob > 0.7 and confidence > 0.7:
            recommended_slice = 2  # High
        elif combat_prob > 0.4 or (confidence < 0.5 and rl_state[6] > 0.5):  # performance_risk
            recommended_slice = 1  # Medium
        else:
            recommended_slice = 0  # Low
        
        return {
            'recommended_slice': recommended_slice,
            'combat_probability': combat_prob,
            'confidence': confidence,
            'urgency': urgency,
            'reason': self._get_recommendation_reason(combat_prob, confidence, rl_state)
        }
    
    def _get_recommendation_reason(self, combat_prob: float, confidence: float, 
                                  state: np.ndarray) -> str:
        """Generate human-readable recommendation reason"""
        if combat_prob > 0.7 and confidence > 0.7:
            return "High probability of combat detected - premium resources needed"
        elif state[2] > 0.5:  # High ping
            return "Network latency exceeds threshold - upgrade recommended"
        elif state[6] > 0.5:  # Performance risk
            return "Performance risk detected - preventive upgrade suggested"
        elif combat_prob < 0.3:
            return "Low activity detected - minimal resources sufficient"
        else:
            return "Moderate activity - balanced resource allocation"


class RLAgentInterface:
    """Interface for Q-learning agent to use predictions"""
    
    def __init__(self, predictor: PredictionIntegration):
        self.predictor = predictor
        self.action_history = deque(maxlen=1000)
        self.performance_metrics = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'proactive_successes': 0,
            'resource_efficiency': []
        }
    
    def get_state_with_prediction(self, game_state: Dict, network_state: Dict) -> Dict:
        """
        Get complete state information for RL agent
        
        Returns:
            Dictionary with state vector and prediction info
        """
        # Get RL state vector
        state_vector = self.predictor.get_rl_state(game_state, network_state)
        
        # Get prediction details
        combat_prob, prediction, confidence = self.predictor.get_prediction(game_state)
        
        # Get recommendation
        recommendation = self.predictor.get_allocation_recommendation(
            state_vector, (combat_prob, prediction, confidence)
        )
        
        return {
            'state_vector': state_vector,
            'prediction': {
                'combat_probability': combat_prob,
                'binary_prediction': prediction,
                'confidence': confidence
            },
            'recommendation': recommendation,
            'timestamp': time.time()
        }
    
    def record_action(self, state_info: Dict, action: int, reward: float):
        """Record action taken by RL agent for analysis"""
        self.action_history.append({
            'timestamp': state_info['timestamp'],
            'state': state_info['state_vector'].tolist(),
            'prediction': state_info['prediction'],
            'recommendation': state_info['recommendation'],
            'action_taken': action,
            'reward': reward
        })
        
        # Update metrics
        if state_info['recommendation']['recommended_slice'] == action:
            self.performance_metrics['correct_predictions'] += 1
        self.performance_metrics['total_predictions'] += 1
    
    def get_performance_summary(self) -> Dict:
        """Get summary of integration performance"""
        if self.performance_metrics['total_predictions'] == 0:
            accuracy = 0
        else:
            accuracy = (self.performance_metrics['correct_predictions'] / 
                       self.performance_metrics['total_predictions'])
        
        # Analyze recent actions
        recent_actions = list(self.action_history)[-100:]
        avg_reward = np.mean([a['reward'] for a in recent_actions]) if recent_actions else 0
        
        # Resource efficiency (how often we use high slice)
        high_slice_usage = sum(1 for a in recent_actions if a['action_taken'] == 2) / len(recent_actions) if recent_actions else 0
        
        return {
            'prediction_accuracy': accuracy,
            'average_reward': avg_reward,
            'high_slice_usage': high_slice_usage,
            'total_decisions': self.performance_metrics['total_predictions'],
            'integration_status': 'operational'
        }


# Example usage and testing
def test_integration():
    """Test the integration components"""
    print("🧪 Testing Model-RL Integration")
    print("=" * 50)
    
    # Initialize predictor
    predictor = PredictionIntegration(use_api=False)
    rl_interface = RLAgentInterface(predictor)
    
    # Simulate game states
    test_states = [
        # Low activity
        {
            'game_state': {
                'mouse_speed': 150,
                'turning_rate': 100,
                'movement_keys': 1,
                'is_shooting': 0,
                'activity_score': 0.2,
                'mouse_speed_ma': 140,
                'activity_ma': 0.18,
                'combat_likelihood': 0.15
            },
            'network_state': {
                'ping_ms': 45,
                'network_quality': 0.8,
                'system_stress': 0.3,
                'performance_risk': 0.1
            }
        },
        # High activity (combat)
        {
            'game_state': {
                'mouse_speed': 850,
                'turning_rate': 620,
                'movement_keys': 3,
                'is_shooting': 1,
                'activity_score': 0.85,
                'mouse_speed_ma': 800,
                'activity_ma': 0.82,
                'combat_likelihood': 0.90
            },
            'network_state': {
                'ping_ms': 75,
                'network_quality': 0.6,
                'system_stress': 0.7,
                'performance_risk': 0.4
            }
        }
    ]
    
    # Test each state
    for i, test in enumerate(test_states):
        print(f"\n📊 Test Case {i+1}:")
        print("-" * 30)
        
        # Get state with prediction
        state_info = rl_interface.get_state_with_prediction(
            test['game_state'], 
            test['network_state']
        )
        
        # Display results
        print(f"State Vector: {state_info['state_vector']}")
        print(f"Combat Probability: {state_info['prediction']['combat_probability']:.3f}")
        print(f"Confidence: {state_info['prediction']['confidence']:.3f}")
        print(f"Recommended Slice: {state_info['recommendation']['recommended_slice']}")
        print(f"Reason: {state_info['recommendation']['reason']}")
        
        # Simulate RL agent action
        action = state_info['recommendation']['recommended_slice']
        reward = 1.0 if action == state_info['recommendation']['recommended_slice'] else -0.5
        
        rl_interface.record_action(state_info, action, reward)
    
    # Show performance summary
    print("\n📈 Integration Performance Summary:")
    print("-" * 40)
    summary = rl_interface.get_performance_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_integration()