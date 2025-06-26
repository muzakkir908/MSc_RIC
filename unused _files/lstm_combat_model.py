import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import os
import json
from datetime import datetime

class LightweightLSTM(nn.Module):
    """Lightweight LSTM for edge deployment - optimized for low latency"""
    def __init__(self, input_size, hidden_size=24, num_layers=1, dropout=0.2):
        super(LightweightLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer - kept small for edge deployment
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers for combat prediction
        self.fc1 = nn.Linear(hidden_size, 16)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 2)  # Binary classification (combat/non-combat)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use only the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class CombatPredictionDataset(Dataset):
    """Custom dataset for sequence-based combat prediction"""
    def __init__(self, features, labels, sequence_length=30):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        sequence = self.features[idx:idx+self.sequence_length]
        # Get label at the end of sequence (predicting future combat)
        label = self.labels[idx+self.sequence_length]
        return torch.FloatTensor(sequence), torch.LongTensor([label])

class ModelTrainer:
    def __init__(self, data_path):
        """Initialize trainer with processed data"""
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load and prepare data
        self.load_data()
        
    def load_data(self):
        """Load the processed gaming data"""
        print("\n📊 Loading processed data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(self.df):,} data points")
        
        # Select features for combat prediction
        self.feature_columns = [
            'mouse_speed',
            'turning_rate',
            'movement_keys',
            'is_shooting',
            'activity_score',
            'mouse_speed_ma',
            'activity_ma',
            'combat_likelihood'
        ]
        
        # Ensure all features exist
        self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
        print(f"📌 Using {len(self.feature_columns)} features: {self.feature_columns}")
        
    def prepare_sequences(self, sequence_length=30):
        """Prepare sequential data for LSTM"""
        print(f"\n🔄 Preparing sequences (length={sequence_length})...")
        
        # Extract features and labels
        features = self.df[self.feature_columns].values
        labels = self.df['is_combat'].values
        
        # Normalize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(features_scaled) - sequence_length):
            X_sequences.append(features_scaled[i:i+sequence_length])
            # Predict combat 2.5 seconds ahead (25 timesteps at 100ms each)
            future_idx = min(i + sequence_length + 25, len(labels) - 1)
            y_sequences.append(labels[future_idx])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"✅ Created {len(X_sequences)} sequences")
        print(f"   Shape: {X_sequences.shape}")
        print(f"   Combat ratio: {y_sequences.mean():.2%}")
        
        return X_sequences, y_sequences
    
    def create_data_loaders(self, X, y, batch_size=64, val_split=0.2):
        """Create train and validation data loaders"""
        # Temporal split (not random) for realistic evaluation
        split_idx = int(len(X) * (1 - val_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        print(f"\n📦 Data loaders created:")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        print(f"   Batch size: {batch_size}")
        
        return train_loader, val_loader
    
    def train_model(self, model, train_loader, val_loader, epochs=50, lr=0.001):
        """Train the LSTM model"""
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 2.0]).to(self.device))  # Weight combat class higher
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': []
        }
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_start = time.time()
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.squeeze())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.squeeze())
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', pos_label=1
            )
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_precision'].append(precision)
            history['val_recall'].append(recall)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model_state = model.state_dict().copy()
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                elapsed = time.time() - train_start
                print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s)")
                print(f"  Loss - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
                print(f"  Accuracy: {val_accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.3f}")
        
        return history
    
    def evaluate_model(self, model, val_loader):
        """Comprehensive model evaluation"""
        print("\n📊 Evaluating model performance...")
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Combat probability
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', pos_label=1
        )
        
        print(f"\n🎯 Model Performance:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\n📈 Confusion Matrix:")
        print(f"   True Negatives: {cm[0,0]:,}")
        print(f"   False Positives: {cm[0,1]:,}")
        print(f"   False Negatives: {cm[1,0]:,}")
        print(f"   True Positives: {cm[1,1]:,}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def visualize_results(self, history, eval_results):
        """Create comprehensive visualizations"""
        print("\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Training history - Loss
        ax = axes[0, 0]
        ax.plot(history['train_loss'], label='Train Loss')
        ax.plot(history['val_loss'], label='Val Loss')
        ax.set_title('Model Loss During Training')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Training history - Accuracy
        ax = axes[0, 1]
        ax.plot(history['val_accuracy'], label='Accuracy', color='green')
        ax.plot(history['val_precision'], label='Precision', color='blue')
        ax.plot(history['val_recall'], label='Recall', color='red')
        ax.set_title('Validation Metrics During Training')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix
        ax = axes[0, 2]
        cm = eval_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['Non-Combat', 'Combat'])
        ax.set_yticklabels(['Non-Combat', 'Combat'])
        
        # 4. Prediction distribution
        ax = axes[1, 0]
        probs = eval_results['probabilities']
        labels = eval_results['labels']
        
        combat_probs = [p for p, l in zip(probs, labels) if l == 1]
        noncombat_probs = [p for p, l in zip(probs, labels) if l == 0]
        
        ax.hist(noncombat_probs, bins=30, alpha=0.5, label='Non-Combat', color='blue')
        ax.hist(combat_probs, bins=30, alpha=0.5, label='Combat', color='red')
        ax.set_title('Prediction Probability Distribution')
        ax.set_xlabel('Combat Probability')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Performance by threshold
        ax = axes[1, 1]
        thresholds = np.linspace(0, 1, 50)
        precisions = []
        recalls = []
        
        for thresh in thresholds:
            preds = (np.array(probs) > thresh).astype(int)
            if preds.sum() > 0:  # Avoid division by zero
                prec = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)[0]
                rec = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)[1]
                precisions.append(prec)
                recalls.append(rec)
            else:
                precisions.append(0)
                recalls.append(0)
        
        ax.plot(thresholds, precisions, label='Precision', color='blue')
        ax.plot(thresholds, recalls, label='Recall', color='red')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Performance by Decision Threshold')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Model summary
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""Model Performance Summary
        
Accuracy: {eval_results['accuracy']:.3f}
Precision: {eval_results['precision']:.3f}
Recall: {eval_results['recall']:.3f}
F1-Score: {eval_results['f1_score']:.3f}

Total Predictions: {len(eval_results['predictions']):,}
Combat Events: {sum(eval_results['labels']):,}

Model Size: 24 LSTM units
Sequence Length: 30 (3 seconds)
Prediction Horizon: 2.5 seconds"""
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('lstm_model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model, eval_results, output_dir='model_output'):
        """Save model and associated files for deployment"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Saving model to {output_dir}/...")
        
        # Save PyTorch model
        model_path = os.path.join(output_dir, 'combat_prediction_lstm.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': len(self.feature_columns),
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers
            },
            'feature_columns': self.feature_columns,
            'performance': {
                'accuracy': eval_results['accuracy'],
                'precision': eval_results['precision'],
                'recall': eval_results['recall'],
                'f1_score': eval_results['f1_score']
            }
        }, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved: {scaler_path}")
        
        # Save model info for deployment
        model_info = {
            'model_type': 'LSTM',
            'input_features': self.feature_columns,
            'sequence_length': 30,
            'prediction_horizon_ms': 2500,
            'model_size_kb': os.path.getsize(model_path) / 1024,
            'performance_metrics': {
                'accuracy': float(eval_results['accuracy']),
                'precision': float(eval_results['precision']),
                'recall': float(eval_results['recall']),
                'f1_score': float(eval_results['f1_score'])
            },
            'training_info': {
                'total_samples': len(self.df),
                'training_date': datetime.now().isoformat(),
                'device': str(self.device)
            }
        }
        
        info_path = os.path.join(output_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✅ Model info saved: {info_path}")
        
        # Create inference script
        self.create_inference_script(output_dir)
        
        return model_path
    
    def create_inference_script(self, output_dir):
        """Create a standalone inference script for deployment"""
        inference_code = '''import torch
import torch.nn as nn
import numpy as np
import pickle
import json

class LightweightLSTM(nn.Module):
    """Lightweight LSTM for edge deployment"""
    def __init__(self, input_size, hidden_size=24, num_layers=1):
        super(LightweightLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        last_output = lstm_out[:, -1, :]
        
        out = self.relu(self.fc1(last_output))
        out = self.fc2(out)
        
        return out

class CombatPredictor:
    def __init__(self, model_path='combat_prediction_lstm.pth', scaler_path='feature_scaler.pkl'):
        # Load model config and weights
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['model_config']
        
        # Initialize model
        self.model = LightweightLSTM(
            config['input_size'],
            config['hidden_size'],
            config['num_layers']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Store config
        self.feature_columns = checkpoint['feature_columns']
        self.sequence_buffer = []
        
    def predict(self, features_dict):
        """Predict combat probability from current features"""
        # Extract features in correct order
        features = [features_dict.get(col, 0) for col in self.feature_columns]
        
        # Scale features
        features_scaled = self.scaler.transform([features])[0]
        
        # Add to buffer
        self.sequence_buffer.append(features_scaled)
        
        # Keep only last 30 timesteps
        if len(self.sequence_buffer) > 30:
            self.sequence_buffer.pop(0)
        
        # Need full sequence for prediction
        if len(self.sequence_buffer) < 30:
            return 0.0, 0  # Not enough data yet
        
        # Convert to tensor
        sequence = torch.FloatTensor([self.sequence_buffer])
        
        # Predict
        with torch.no_grad():
            output = self.model(sequence)
            probs = torch.softmax(output, dim=1)
            combat_prob = probs[0, 1].item()
            prediction = 1 if combat_prob > 0.5 else 0
        
        return combat_prob, prediction

# Example usage
if __name__ == "__main__":
    predictor = CombatPredictor()
    
    # Simulate real-time prediction
    example_features = {
        'mouse_speed': 450.0,
        'turning_rate': 320.0,
        'movement_keys': 2,
        'is_shooting': 1,
        'activity_score': 0.65,
        'mouse_speed_ma': 420.0,
        'activity_ma': 0.60,
        'combat_likelihood': 0.70
    }
    
    prob, pred = predictor.predict(example_features)
    print(f"Combat probability: {prob:.3f}")
    print(f"Prediction: {'COMBAT' if pred == 1 else 'NON-COMBAT'}")
'''
        
        script_path = os.path.join(output_dir, 'inference.py')
        with open(script_path, 'w') as f:
            f.write(inference_code)
        print(f"✅ Inference script saved: {script_path}")

def main():
    """Main training pipeline"""
    print("🎮 LSTM COMBAT PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Get data path
    data_path = input("Enter path to processed gaming data CSV: ").strip()
    if not data_path:
        print("❌ Please provide data path")
        return
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(data_path)
        
        # Prepare sequences
        X, y = trainer.prepare_sequences(sequence_length=30)
        
        # Create data loaders
        train_loader, val_loader = trainer.create_data_loaders(X, y, batch_size=64)
        
        # Initialize model
        input_size = len(trainer.feature_columns)
        model = LightweightLSTM(input_size=input_size, hidden_size=24, num_layers=1)
        model = model.to(trainer.device)
        
        print(f"\n🏗️ Model Architecture:")
        print(f"   Input features: {input_size}")
        print(f"   LSTM units: 24")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        history = trainer.train_model(model, train_loader, val_loader, epochs=50)
        
        # Evaluate model
        eval_results = trainer.evaluate_model(model, val_loader)
        
        # Visualize results
        trainer.visualize_results(history, eval_results)
        
        # Save model
        model_path = trainer.save_model(model, eval_results)
        
        print("\n✅ TRAINING COMPLETE!")
        print(f"📊 Final Performance:")
        print(f"   Accuracy: {eval_results['accuracy']:.3f}")
        print(f"   Precision: {eval_results['precision']:.3f}")  
        print(f"   Recall: {eval_results['recall']:.3f}")
        print(f"   F1-Score: {eval_results['f1_score']:.3f}")
        
        # Check against success criteria
        if eval_results['accuracy'] >= 0.75:
            print("\n✅ SUCCESS: Model meets accuracy threshold (≥75%)")
        else:
            print(f"\n⚠️ Model accuracy ({eval_results['accuracy']:.1%}) below target (75%)")
            print("   Consider: More data, hyperparameter tuning, or feature engineering")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()