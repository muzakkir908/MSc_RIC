import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pickle
import json
from datetime import datetime

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

class LSTMModel(nn.Module):
    """LSTM for combat prediction"""
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

class CombatDataset(Dataset):
    """Dataset for sequence data"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def load_and_prepare_data(file_path):
    """Load and prepare the data"""
    print("📊 Loading data...")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df):,} data points")
    
    # Analyze class distribution
    combat_ratio = df['is_combat'].mean()
    print(f"📈 Class distribution: {combat_ratio:.1%} combat, {1-combat_ratio:.1%} non-combat")
    
    # Select features
    feature_columns = [
        'mouse_speed', 'turning_rate', 'movement_keys', 
        'is_shooting', 'activity_score', 'keys_pressed',
        'ping_ms', 'cpu_percent'
    ]
    
    # Add optional features
    optional_features = ['mouse_speed_ma', 'activity_ma', 'combat_likelihood']
    for feat in optional_features:
        if feat in df.columns:
            feature_columns.append(feat)
    
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"📌 Using {len(feature_columns)} features")
    
    return df, feature_columns

def create_balanced_sequences(df, feature_columns, sequence_length=30):
    """Create balanced sequences for training"""
    print(f"\n🔄 Creating balanced sequences...")
    
    # Extract features and labels
    features = df[feature_columns].fillna(0).values
    labels = df['is_combat'].values
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    combat_sequences = []
    noncombat_sequences = []
    
    for i in range(len(features_scaled) - sequence_length - 20):
        sequence = features_scaled[i:i+sequence_length]
        
        # Look ahead 2 seconds (20 timesteps)
        future_window = labels[i+sequence_length:i+sequence_length+20]
        has_combat = np.any(future_window)
        
        if has_combat:
            combat_sequences.append((sequence, 1))
        else:
            noncombat_sequences.append((sequence, 0))
    
    print(f"   Raw sequences - Combat: {len(combat_sequences)}, Non-combat: {len(noncombat_sequences)}")
    
    # Balance the dataset
    min_class = min(len(combat_sequences), len(noncombat_sequences))
    
    # Sample equally from both classes
    combat_sequences = combat_sequences[:min_class]
    noncombat_sequences = noncombat_sequences[:min_class]
    
    # Combine and shuffle
    all_sequences = combat_sequences + noncombat_sequences
    np.random.shuffle(all_sequences)
    
    # Extract X and y
    X = np.array([seq[0] for seq in all_sequences])
    y = np.array([seq[1] for seq in all_sequences])
    
    print(f"✅ Created balanced dataset:")
    print(f"   Total sequences: {len(X)}")
    print(f"   Combat: {y.sum()} ({y.mean():.1%})")
    print(f"   Non-combat: {len(y) - y.sum()} ({1-y.mean():.1%})")
    
    return X, y, scaler

def create_weighted_sampler(y_train):
    """Create weighted sampler for imbalanced data"""
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    weights = class_weights[y_train]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler

def train_model(model, train_loader, val_loader, epochs=50, device='cpu', class_weights=None):
    """Train the LSTM model with class weights"""
    print(f"\n🚀 Training model for {epochs} epochs...")
    
    # Use weighted loss if provided
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    best_f1 = 0
    best_model = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(batch_y.numpy())
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        
        # Calculate F1 score for combat class
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, pos_label=1)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        
        # Update learning rate
        scheduler.step(avg_train_loss)
        
        # Save best model based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {avg_train_loss:.4f}")
            print(f"  Accuracy: {val_accuracy:.3f}, F1 Score: {val_f1:.3f}")
    
    # Load best model
    model.load_state_dict(best_model)
    print(f"\n✅ Training complete! Best F1 score: {best_f1:.3f}")
    
    return train_losses, val_accuracies, val_f1_scores

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the trained model"""
    print("\n📊 Evaluating model...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Combat probability
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Test Accuracy: {accuracy:.3f}")
    
    # Detailed report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Non-Combat', 'Combat']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    return accuracy, all_preds, all_labels, all_probs

def save_model(model, scaler, feature_columns, performance, save_dir='trained_model'):
    """Save the trained model"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': len(feature_columns),
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers
        },
        'feature_columns': feature_columns,
        'performance': performance
    }, f'{save_dir}/lstm_model.pth')
    
    # Save scaler
    with open(f'{save_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save config
    config = {
        'feature_columns': feature_columns,
        'sequence_length': 30,
        'model_type': 'LSTM',
        'training_date': datetime.now().isoformat(),
        'performance': performance
    }
    
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Model saved to '{save_dir}/' directory")

def plot_results(train_losses, val_accuracies, val_f1_scores):
    """Plot training results"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(val_accuracies)
    ax2.axhline(y=0.75, color='r', linestyle='--', label='Target (75%)')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # F1 Score plot
    ax3.plot(val_f1_scores)
    ax3.set_title('Validation F1 Score (Combat)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def main():
    """Main training function"""
    print("🎮 BALANCED LSTM TRAINING FOR COMBAT PREDICTION")
    print("=" * 60)
    
    # Get data file path
    data_path = input("Enter path to your processed data CSV file: ").strip()
    
    # Load data
    df, feature_columns = load_and_prepare_data(data_path)
    
    # Create balanced sequences
    X, y, scaler = create_balanced_sequences(df, feature_columns, sequence_length=30)
    
    # Split data
    n_samples = len(X)
    train_idx = int(0.8 * n_samples)
    val_idx = int(0.9 * n_samples)
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(X_train):,} sequences ({y_train.mean():.1%} combat)")
    print(f"   Val: {len(X_val):,} sequences ({y_val.mean():.1%} combat)")
    print(f"   Test: {len(X_test):,} sequences ({y_test.mean():.1%} combat)")
    
    # Calculate class weights
    classes = np.array([0, 1])
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.FloatTensor(class_weights_array)
    print(f"\n⚖️ Class weights: {class_weights_array}")
    
    # Create datasets and loaders
    train_dataset = CombatDataset(X_train, y_train)
    val_dataset = CombatDataset(X_val, y_val)
    test_dataset = CombatDataset(X_test, y_test)
    
    # Use weighted sampler for training
    train_sampler = create_weighted_sampler(y_train)
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    input_size = len(feature_columns)
    model = LSTMModel(input_size=input_size, hidden_size=32, num_layers=2, dropout=0.3)
    model = model.to(device)
    
    print(f"\n🏗️ Model initialized:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_accuracies, val_f1_scores = train_model(
        model, train_loader, val_loader, epochs=50, device=device, class_weights=class_weights
    )
    
    # Evaluate on test set
    test_accuracy, test_preds, test_labels, test_probs = evaluate_model(model, test_loader, device=device)
    
    # Save model
    performance = {
        'test_accuracy': float(test_accuracy),
        'best_val_accuracy': float(max(val_accuracies)),
        'best_val_f1': float(max(val_f1_scores))
    }
    save_model(model, scaler, feature_columns, performance)
    
    # Plot results
    plot_results(train_losses, val_accuracies, val_f1_scores)
    
    print("\n✅ Training complete!")
    print(f"📊 Final test accuracy: {test_accuracy:.1%}")
    
    if test_accuracy >= 0.75:
        print("🎉 Model meets the 75% accuracy requirement!")
    else:
        print("📈 Model performance improved but consider further tuning.")

if __name__ == "__main__":
    main()