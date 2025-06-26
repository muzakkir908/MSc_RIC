import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import json
from datetime import datetime
import os
from lstm_combat_model import LightweightLSTM, ModelTrainer

class HyperparameterTuner:
    """Hyperparameter tuning for LSTM model"""
    
    def __init__(self, data_path):
        self.trainer = ModelTrainer(data_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define hyperparameter search space
        self.param_grid = {
            'hidden_size': [16, 24, 32],
            'num_layers': [1, 2],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.005, 0.01],
            'batch_size': [32, 64, 128],
            'sequence_length': [20, 30, 40]
        }
        
        self.results = []
        
    def evaluate_params(self, params):
        """Evaluate a single parameter combination"""
        print(f"\n🔧 Testing parameters: {params}")
        
        try:
            # Prepare data with specified sequence length
            X, y = self.trainer.prepare_sequences(sequence_length=params['sequence_length'])
            
            # Create data loaders
            train_loader, val_loader = self.trainer.create_data_loaders(
                X, y, 
                batch_size=params['batch_size'],
                val_split=0.2
            )
            
            # Initialize model with parameters
            input_size = len(self.trainer.feature_columns)
            model = LightweightLSTM(
                input_size=input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            ).to(self.device)
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            
            # Train model (fewer epochs for tuning)
            history = self.trainer.train_model(
                model, 
                train_loader, 
                val_loader,
                epochs=20,  # Reduced for faster tuning
                lr=params['learning_rate']
            )
            
            # Evaluate
            eval_results = self.trainer.evaluate_model(model, val_loader)
            
            # Calculate inference time
            inference_time = self.measure_inference_time(model, val_loader)
            
            # Store results
            result = {
                'params': params,
                'accuracy': eval_results['accuracy'],
                'precision': eval_results['precision'],
                'recall': eval_results['recall'],
                'f1_score': eval_results['f1_score'],
                'final_val_loss': history['val_loss'][-1],
                'param_count': param_count,
                'inference_time_ms': inference_time,
                'training_time': len(history['train_loss']) * 0.5  # Approximate
            }
            
            print(f"✅ Accuracy: {result['accuracy']:.3f}, F1: {result['f1_score']:.3f}, Inference: {result['inference_time_ms']:.1f}ms")
            
            return result
            
        except Exception as e:
            print(f"❌ Error with params {params}: {e}")
            return None
    
    def measure_inference_time(self, model, val_loader):
        """Measure average inference time"""
        model.eval()
        times = []
        
        with torch.no_grad():
            for batch_X, _ in val_loader:
                if len(times) >= 10:  # Measure only 10 batches
                    break
                    
                batch_X = batch_X.to(self.device)
                
                # Warm up
                if len(times) == 0:
                    _ = model(batch_X)
                
                # Measure
                start = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
                end = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
                
                if self.device.type == 'cuda':
                    start.record()
                    _ = model(batch_X)
                    end.record()
                    torch.cuda.synchronize()
                    elapsed = start.elapsed_time(end) / batch_X.size(0)  # Per sample
                else:
                    import time
                    start_time = time.time()
                    _ = model(batch_X)
                    elapsed = (time.time() - start_time) * 1000 / batch_X.size(0)
                
                times.append(elapsed)
        
        return np.mean(times) if times else 0
    
    def grid_search(self, sample_ratio=0.2):
        """Perform grid search on hyperparameters"""
        # For faster tuning, use a sample of parameter combinations
        param_list = list(ParameterGrid(self.param_grid))
        
        if sample_ratio < 1.0:
            n_samples = max(1, int(len(param_list) * sample_ratio))
            param_list = np.random.choice(param_list, n_samples, replace=False)
            print(f"\n🔍 Sampling {n_samples}/{len(list(ParameterGrid(self.param_grid)))} parameter combinations")
        
        print(f"\n🚀 Starting hyperparameter search with {len(param_list)} combinations...")
        
        for i, params in enumerate(param_list):
            print(f"\n📊 Progress: {i+1}/{len(param_list)}")
            result = self.evaluate_params(params)
            if result:
                self.results.append(result)
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze tuning results and find best parameters"""
        if not self.results:
            print("❌ No successful results to analyze")
            return None
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(self.results)
        
        # Find best by different criteria
        best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
        best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
        best_recall = results_df.loc[results_df['recall'].idxmax()]
        
        # Find best for edge deployment (balance accuracy and speed)
        # Edge score = accuracy * 0.7 + (1 - normalized_inference_time) * 0.3
        results_df['inference_norm'] = results_df['inference_time_ms'] / results_df['inference_time_ms'].max()
        results_df['edge_score'] = (
            results_df['accuracy'] * 0.7 + 
            (1 - results_df['inference_norm']) * 0.3
        )
        best_edge = results_df.loc[results_df['edge_score'].idxmax()]
        
        print("\n" + "="*60)
        print("🏆 HYPERPARAMETER TUNING RESULTS")
        print("="*60)
        
        print("\n📊 Top 5 by Accuracy:")
        top_5 = results_df.nlargest(5, 'accuracy')[['params', 'accuracy', 'f1_score', 'inference_time_ms']]
        for idx, row in top_5.iterrows():
            print(f"  {row['params']} → Acc: {row['accuracy']:.3f}, F1: {row['f1_score']:.3f}, Time: {row['inference_time_ms']:.1f}ms")
        
        print(f"\n🎯 Best Overall (Accuracy): {best_accuracy['accuracy']:.3f}")
        print(f"   Parameters: {best_accuracy['params']}")
        print(f"   Model size: {best_accuracy['param_count']:,} parameters")
        print(f"   Inference time: {best_accuracy['inference_time_ms']:.1f}ms")
        
        print(f"\n⚡ Best for Edge Deployment: {best_edge['accuracy']:.3f}")
        print(f"   Parameters: {best_edge['params']}")
        print(f"   Model size: {best_edge['param_count']:,} parameters")
        print(f"   Inference time: {best_edge['inference_time_ms']:.1f}ms")
        print(f"   Edge score: {best_edge['edge_score']:.3f}")
        
        # Save results
        self.save_results(results_df, best_edge)
        
        return best_edge['params']
    
    def save_results(self, results_df, best_params):
        """Save tuning results"""
        output_dir = 'tuning_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results
        results_df.to_csv(os.path.join(output_dir, 'tuning_results.csv'), index=False)
        
        # Save best parameters
        best_config = {
            'best_params': dict(best_params['params']),
            'performance': {
                'accuracy': float(best_params['accuracy']),
                'f1_score': float(best_params['f1_score']),
                'inference_time_ms': float(best_params['inference_time_ms']),
                'param_count': int(best_params['param_count'])
            },
            'tuning_date': datetime.now().isoformat(),
            'total_experiments': len(results_df)
        }
        
        with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\n💾 Results saved to {output_dir}/")
    
    def train_final_model(self, best_params):
        """Train final model with best parameters"""
        print("\n🚀 Training final model with best parameters...")
        
        # Prepare data
        X, y = self.trainer.prepare_sequences(sequence_length=best_params['sequence_length'])
        train_loader, val_loader = self.trainer.create_data_loaders(
            X, y, 
            batch_size=best_params['batch_size']
        )
        
        # Initialize model
        input_size = len(self.trainer.feature_columns)
        model = LightweightLSTM(
            input_size=input_size,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout']
        ).to(self.device)
        
        # Train with more epochs
        history = self.trainer.train_model(
            model,
            train_loader,
            val_loader,
            epochs=50,
            lr=best_params['learning_rate']
        )
        
        # Final evaluation
        eval_results = self.trainer.evaluate_model(model, val_loader)
        
        # Visualize
        self.trainer.visualize_results(history, eval_results)
        
        # Save
        self.trainer.save_model(model, eval_results, output_dir='model_output_tuned')
        
        return model, eval_results

def main():
    """Main tuning pipeline"""
    print("🔧 HYPERPARAMETER TUNING FOR LSTM MODEL")
    print("=" * 60)
    
    data_path = input("Enter path to processed gaming data CSV: ").strip()
    if not data_path:
        print("❌ Please provide data path")
        return
    
    try:
        # Initialize tuner
        tuner = HyperparameterTuner(data_path)
        
        # Perform grid search
        best_params = tuner.grid_search(sample_ratio=0.3)  # Sample 30% for faster tuning
        
        if best_params:
            # Train final model
            print("\n" + "="*60)
            print("🎯 TRAINING FINAL MODEL")
            print("="*60)
            
            model, eval_results = tuner.train_final_model(best_params)
            
            print("\n✅ TUNING COMPLETE!")
            print(f"📊 Final Model Performance:")
            print(f"   Accuracy: {eval_results['accuracy']:.3f}")
            print(f"   Precision: {eval_results['precision']:.3f}")
            print(f"   Recall: {eval_results['recall']:.3f}")
            print(f"   F1-Score: {eval_results['f1_score']:.3f}")
            
    except Exception as e:
        print(f"❌ Error during tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()