import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimpleBaseline:
    """Simple ML baseline to validate if the problem is with LSTM or data"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.load_data()
        
    def load_data(self):
        """Load and prepare data"""
        print("📊 Loading data for baseline models...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df):,} data points")
        
        # Use simple features that should work
        self.features = [
            'mouse_speed',
            'turning_rate',
            'is_shooting',
            'movement_keys',
            'activity_score'
        ]
        
        # Add any available additional features
        optional_features = ['mouse_speed_ma', 'combat_likelihood', 'keys_pressed']
        for feat in optional_features:
            if feat in self.df.columns:
                self.features.append(feat)
                
        print(f"Using {len(self.features)} features: {self.features}")
        
    def create_simple_features(self):
        """Create simple but effective features"""
        print("\n🔧 Creating simple features...")
        
        # Basic rolling features
        for window in [10, 30]:
            self.df[f'mouse_active_{window}'] = (
                self.df['mouse_speed'].rolling(window, min_periods=1).mean() > 100
            ).astype(int)
            
            self.df[f'high_activity_{window}'] = (
                self.df['activity_score'].rolling(window, min_periods=1).mean() > 0.5
            ).astype(int)
        
        # Simple interaction
        self.df['intense_movement'] = (
            (self.df['mouse_speed'] > 500) & 
            (self.df['turning_rate'] > 300)
        ).astype(int)
        
        # Add new features
        new_features = [col for col in self.df.columns if col.startswith(('mouse_active_', 'high_activity_', 'intense_'))]
        self.features.extend(new_features)
        
        print(f"Added {len(new_features)} new features")
        
    def prepare_train_test_split(self, test_size=0.2):
        """Prepare data for training"""
        # Extract features and labels
        X = self.df[self.features].fillna(0).values
        y = self.df['is_combat'].values
        
        # Use last 20% for testing (temporal split)
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n📊 Data split:")
        print(f"   Training: {len(self.X_train):,} samples")
        print(f"   Testing: {len(self.X_test):,} samples")
        print(f"   Combat ratio (train): {self.y_train.mean():.1%}")
        print(f"   Combat ratio (test): {self.y_test.mean():.1%}")
        
    def train_baseline_models(self):
        """Train multiple baseline models"""
        print("\n🚀 Training baseline models...")
        
        # Define models to test
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train
            if name == 'Logistic Regression':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Evaluate
            print(f"\n{name} Results:")
            print("-" * 50)
            print(classification_report(self.y_test, y_pred, 
                                      target_names=['Non-Combat', 'Combat'],
                                      digits=3))
            
            # Store model and results
            self.models[name] = model
            results[name] = {
                'predictions': y_pred,
                'report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\nTop 5 Important Features:")
                for _, row in importance_df.head().iterrows():
                    print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return results
    
    def analyze_errors(self, model_name='Random Forest'):
        """Analyze where the model makes mistakes"""
        print(f"\n🔍 Error Analysis for {model_name}:")
        print("-" * 50)
        
        model = self.models[model_name]
        
        # Get predictions
        if model_name == 'Logistic Regression':
            y_pred = model.predict(self.X_test_scaled)
            y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Find errors
        errors = y_pred != self.y_test
        false_negatives = errors & (self.y_test == 1)  # Missed combat
        false_positives = errors & (self.y_test == 0)  # False alarms
        
        print(f"Total errors: {errors.sum()} ({errors.mean()*100:.1f}%)")
        print(f"False negatives (missed combat): {false_negatives.sum()}")
        print(f"False positives (false alarms): {false_positives.sum()}")
        
        # Analyze error patterns
        test_features = self.X_test[errors]
        correct_features = self.X_test[~errors]
        
        print("\nFeature averages for errors vs correct:")
        for i, feat in enumerate(self.features[:5]):  # First 5 features
            error_avg = test_features[:, i].mean()
            correct_avg = correct_features[:, i].mean()
            print(f"  {feat}: Error={error_avg:.2f}, Correct={correct_avg:.2f}")
        
        # Confidence analysis
        print("\nPrediction confidence:")
        print(f"  Correct predictions: {y_proba[~errors].mean():.3f} avg confidence")
        print(f"  Errors: {y_proba[errors].mean():.3f} avg confidence")
        
        return errors, y_proba
    
    def save_best_model(self, model_name='Random Forest'):
        """Save the best performing model"""
        import os
        os.makedirs('baseline_model', exist_ok=True)
        
        model = self.models[model_name]
        
        # Save model
        if model_name == 'Logistic Regression':
            joblib.dump(model, 'baseline_model/baseline_lr.pkl')
            joblib.dump(self.scaler, 'baseline_model/baseline_scaler.pkl')
        else:
            joblib.dump(model, 'baseline_model/baseline_rf.pkl')
        
        # Save feature list
        with open('baseline_model/features.txt', 'w') as f:
            for feat in self.features:
                f.write(f"{feat}\n")
        
        print(f"\n✅ Saved {model_name} model to 'baseline_model/' directory")

def main():
    """Run baseline comparison"""
    print("🎯 BASELINE MODEL COMPARISON")
    print("=" * 60)
    print("This will test if simple models work better than LSTM\n")
    
    data_path = input("Enter path to processed gaming data CSV: ").strip()
    if not data_path:
        print("No path provided")
        return
    
    try:
        # Initialize
        baseline = SimpleBaseline(data_path)
        
        # Create features
        baseline.create_simple_features()
        
        # Prepare data
        baseline.prepare_train_test_split()
        
        # Train models
        results = baseline.train_baseline_models()
        
        # Find best model
        best_model = None
        best_f1 = 0
        
        for name, result in results.items():
            f1 = result['report']['1']['f1-score']
            accuracy = result['report']['accuracy']
            
            print(f"\n{name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = name
        
        print(f"\n🏆 Best model: {best_model} with F1={best_f1:.3f}")
        
        # Error analysis on best model
        baseline.analyze_errors(best_model)
        
        # Save best model
        baseline.save_best_model(best_model)
        
        # Recommendations
        print("\n📋 RECOMMENDATIONS:")
        print("-" * 50)
        
        best_accuracy = results[best_model]['report']['accuracy']
        
        if best_accuracy > 0.75:
            print("✅ Baseline achieves target accuracy!")
            print("   → The data is good, LSTM might be overfitting")
            print("   → Try simpler LSTM architecture or use this baseline")
        elif best_accuracy > 0.65:
            print("⚠️ Baseline is close but not quite there")
            print("   → Data quality might be an issue")
            print("   → Consider collecting more distinct combat examples")
            print("   → Try ensemble methods combining LSTM + baseline")
        else:
            print("❌ Even simple models struggle")
            print("   → Data labeling might have issues")
            print("   → Combat definition might be unclear")
            print("   → Consider re-examining the data collection process")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()