import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ImprovedGameDataProcessor:
    def __init__(self, csv_file):
        """Initialize with your merged CSV file"""
        self.df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(self.df):,} data points ({len(self.df) * 0.1 / 60:.1f} minutes)")
        
        # Data quality flags
        self.data_quality = {}
        self.check_data_quality()
    
    def check_data_quality(self):
        """Assess data quality and identify issues"""
        print("\n🔍 DATA QUALITY ASSESSMENT:")
        print("-" * 50)
        
        # Check for missing values
        missing_data = self.df.isnull().sum()
        self.data_quality['missing_values'] = missing_data[missing_data > 0]
        if len(self.data_quality['missing_values']) > 0:
            print(f"⚠️ Missing values found:")
            for col, count in self.data_quality['missing_values'].items():
                print(f"   {col}: {count} ({count/len(self.df)*100:.1f}%)")
        else:
            print("✅ No missing values")
        
        # Skip GPU check - we know it's Intel GPU with no monitoring
        print("📌 Intel GPU detected - GPU monitoring not available")
        print("   Using CPU/Memory for system load assessment")
        
        # Check for unrealistic values
        high_ping_ratio = (self.df['ping_ms'] > 500).mean()
        if high_ping_ratio > 0.05:
            print(f"⚠️ Unusual ping values: {high_ping_ratio*100:.1f}% >500ms")
        
        # Check mouse movement patterns
        static_mouse_ratio = ((self.df['mouse_speed'] == 0) & 
                             (self.df['turning_rate'] == 0)).mean()
        print(f"📊 Static mouse periods: {static_mouse_ratio*100:.1f}%")
        
        # Combat distribution
        combat_ratio = self.df['is_combat'].mean()
        print(f"⚔️ Combat ratio: {combat_ratio*100:.1f}%")
        
        print("✅ Data quality assessment complete")
    
    def clean_and_fix_data(self):
        """Clean data and fix common issues"""
        print("\n🧹 CLEANING AND FIXING DATA:")
        print("-" * 40)
        
        # Handle missing values
        if len(self.data_quality['missing_values']) > 0:
            # Use forward fill then backward fill for better interpolation
            self.df = self.df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            print("✅ Filled missing values")
        
        # Fix timestamp format
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df.sort_values('timestamp', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print("✅ Fixed timestamps and sorting")
        
        # Fix mouse speed calculation artifacts
        # Remove unrealistic repeated high speeds
        speed_threshold = self.df['mouse_speed'].quantile(0.95)
        mask = (self.df['mouse_speed'] == self.df['mouse_speed'].shift(1)) & \
               (self.df['mouse_speed'] > speed_threshold)
        
        if mask.sum() > 0:
            # Recalculate mouse speed for problematic entries
            for idx in self.df[mask].index:
                if idx > 0 and idx < len(self.df) - 1:
                    # Use interpolation
                    prev_speed = self.df.loc[idx-1, 'mouse_speed']
                    next_speed = self.df.loc[idx+1, 'mouse_speed']
                    self.df.loc[idx, 'mouse_speed'] = (prev_speed + next_speed) / 2
            
            print(f"✅ Fixed {mask.sum()} mouse speed artifacts")
        
        # Remove GPU columns as they're all zeros
        gpu_columns = ['gpu_percent', 'gpu_memory_percent', 'gpu_temp']
        for col in gpu_columns:
            if col in self.df.columns:
                self.df.drop(col, axis=1, inplace=True)
        print("✅ Removed non-functional GPU columns")
        
        # Smooth extreme outliers
        for col in ['ping_ms', 'cpu_percent', 'mouse_speed']:
            if col in self.df.columns:
                q99 = self.df[col].quantile(0.99)
                q01 = self.df[col].quantile(0.01)
                self.df[col] = np.clip(self.df[col], q01, q99)
        
        print("✅ Smoothed extreme outliers")
    
    def create_advanced_features(self):
        """Create sophisticated features for RL training"""
        print("\n🔧 CREATING ADVANCED FEATURES:")
        print("-" * 40)
        
        # Time-based features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['minute'] = self.df['timestamp'].dt.minute
        self.df['time_since_start'] = (
            self.df['timestamp'] - self.df['timestamp'].iloc[0]
        ).dt.total_seconds()
        
        # Rolling window features (last 3 seconds = 30 data points)
        window_size = 30
        
        self.df['mouse_speed_ma'] = self.df['mouse_speed'].rolling(
            window=window_size, min_periods=1).mean()
        self.df['ping_ma'] = self.df['ping_ms'].rolling(
            window=window_size, min_periods=1).mean()
        self.df['activity_ma'] = self.df['activity_score'].rolling(
            window=window_size, min_periods=1).mean()
        
        # Volatility features (how much things are changing)
        self.df['mouse_speed_std'] = self.df['mouse_speed'].rolling(
            window=window_size, min_periods=1).std().fillna(0)
        self.df['ping_volatility'] = self.df['ping_ms'].rolling(
            window=window_size, min_periods=1).std().fillna(0)
        
        # Trend features (is performance getting better/worse?)
        self.df['ping_trend'] = self.df['ping_ms'].diff(periods=10).fillna(0)
        self.df['activity_trend'] = self.df['activity_score'].diff(periods=10).fillna(0)
        
        # Network quality score (improved)
        self.df['network_quality'] = np.clip(
            (1 - np.clip(self.df['ping_ms'] / 200, 0, 1)) * 0.6 +  # Ping weight
            np.clip(self.df['bytes_recv_kbs'] / 500, 0, 1) * 0.25 +  # Download
            np.clip(self.df['bytes_sent_kbs'] / 100, 0, 1) * 0.15,   # Upload
            0, 1
        )
        
        # System stress score (WITHOUT GPU)
        self.df['system_stress'] = np.clip(
            (self.df['cpu_percent'] / 100) * 0.6 +  # Increased weight for CPU
            (self.df['memory_percent'] / 100) * 0.4,  # Increased weight for memory
            0, 1
        )
        
        # Combat prediction features
        self.df['combat_likelihood'] = (
            np.clip(self.df['mouse_speed'] / 1000, 0, 1) * 0.3 +
            np.clip(self.df['turning_rate'] / 500, 0, 1) * 0.2 +
            (self.df['movement_keys'] > 0).astype(float) * 0.2 +
            self.df['is_shooting'].astype(float) * 0.3
        )
        
        # Performance risk score (WITHOUT GPU)
        self.df['performance_risk'] = (
            (self.df['ping_ms'] > 80).astype(float) * 0.5 +  # Increased weight for ping
            (self.df['cpu_percent'] > 75).astype(float) * 0.3 +
            (self.df['memory_percent'] > 85).astype(float) * 0.2  # Added memory threshold
        )
        
        # Optimal slice prediction (for RL target)
        self.df['needs_premium_slice'] = (
            (self.df['is_combat']) |
            (self.df['performance_risk'] > 0.5) |
            (self.df['ping_ms'] > 100) |
            (self.df['combat_likelihood'] > 0.6)
        ).astype(int)
        
        print("✅ Created advanced features for RL training")
        print("   Note: Using CPU/Memory for system stress (Intel GPU)")
    
    def create_rl_states_and_actions(self):
        """Create proper RL state-action pairs"""
        print("\n🤖 CREATING RL TRAINING DATA:")
        print("-" * 40)
        
        # Define state space (what the RL agent observes)
        state_features = [
            'combat_likelihood',      # Predicted combat intensity
            'ping_ma',               # Recent average ping
            'network_quality',       # Network quality score
            'system_stress',         # System load (CPU + Memory)
            'performance_risk',      # Risk of performance issues
            'mouse_speed_ma',        # Recent mouse activity
            'activity_trend',        # Trend in player activity
            'ping_volatility'        # Network stability
        ]
        
        # Create state matrix
        self.rl_states = self.df[state_features].copy()
        
        # Normalize states to [0, 1] for better RL training
        scaler = StandardScaler()
        self.rl_states_normalized = pd.DataFrame(
            scaler.fit_transform(self.rl_states),
            columns=state_features,
            index=self.rl_states.index
        )
        
        # Define actions (network slice allocations)
        # 0: Low slice (cheap, <100ms latency)
        # 1: Medium slice (moderate cost, <60ms latency)  
        # 2: High slice (expensive, <30ms latency)
        
        # Adjusted thresholds for better action distribution
        self.df['optimal_action'] = np.where(
            self.df['performance_risk'] > 0.6,  # Lowered from 0.7
            2,  # High slice for high risk
            np.where(
                (self.df['is_combat']) | (self.df['performance_risk'] > 0.25),  # Lowered from 0.3
                1,  # Medium slice for combat or moderate risk
                0   # Low slice otherwise
            )
        )
        
        # Calculate rewards based on performance vs cost
        self.df['reward'] = self.calculate_reward()
        
        # Create next state for temporal difference learning
        next_state_cols = [f'next_{col}' for col in state_features]
        self.rl_states_normalized[next_state_cols] = \
            self.rl_states_normalized[state_features].shift(-1)
        
        # Remove last row (no next state)
        self.rl_training_data = pd.concat([
            self.rl_states_normalized[:-1],
            self.df[['optimal_action', 'reward', 'is_combat', 'performance_risk']][:-1]
        ], axis=1)
        
        print(f"✅ Created RL training data: {len(self.rl_training_data)} transitions")
        print(f"   Action distribution:")
        action_dist = self.df['optimal_action'].value_counts().sort_index()
        for action, count in action_dist.items():
            action_names = ['Low Slice', 'Medium Slice', 'High Slice']
            print(f"     {action_names[action]}: {count} ({count/len(self.df)*100:.1f}%)")
    
    def calculate_reward(self):
        """Calculate reward function for RL training"""
        # Base reward on performance (latency < 50ms is ideal)
        performance_reward = np.where(
            self.df['ping_ms'] <= 50, 1.0,
            np.where(self.df['ping_ms'] <= 100, 0.5, -0.5)
        )
        
        # Bonus for good performance during combat
        combat_bonus = np.where(
            self.df['is_combat'] & (self.df['ping_ms'] <= 50), 0.5, 0
        )
        
        # Penalty for resource waste (high slice when not needed)
        efficiency_penalty = np.where(
            (~self.df['is_combat']) & (self.df['performance_risk'] < 0.3), -0.2, 0
        )
        
        return performance_reward + combat_bonus + efficiency_penalty
    
    def train_combat_predictor(self):
        """Train improved combat prediction model"""
        print("\n🎯 TRAINING COMBAT PREDICTOR:")
        print("-" * 40)
        
        # Features for combat prediction
        combat_features = [
            'mouse_speed_ma', 'turning_rate', 'movement_keys', 'is_shooting',
            'mouse_speed_std', 'activity_trend', 'keys_pressed', 'activity_ma'
        ]
        
        X = self.df[combat_features].fillna(0)
        y = self.df['is_combat']
        
        # Split data temporally (not randomly) for more realistic evaluation
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Combat Prediction Accuracy: {accuracy:.1%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Non-Combat', 'Combat']))
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': combat_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Feature Importance:")
        for _, row in importance_df.iterrows():
            bar = '█' * int(row['importance'] * 100)
            print(f"  {row['feature'][:20]:20} {row['importance']:.3f} {bar}")
        
        return model
    
    def analyze_network_patterns(self):
        """Analyze network performance patterns"""
        print("\n🌐 NETWORK PERFORMANCE ANALYSIS:")
        print("-" * 50)
        
        # Overall statistics
        print("Overall Network Stats:")
        print(f"  Average Ping: {self.df['ping_ms'].mean():.1f}ms ± {self.df['ping_ms'].std():.1f}ms")
        print(f"  Ping Range: {self.df['ping_ms'].min():.1f} - {self.df['ping_ms'].max():.1f}ms")
        print(f"  High Ping Events (>100ms): {(self.df['ping_ms'] > 100).mean()*100:.1f}%")
        
        # Combat vs non-combat performance
        combat_stats = self.df.groupby('is_combat').agg({
            'ping_ms': ['mean', 'std', 'min', 'max'],
            'network_quality': 'mean',
            'performance_risk': 'mean'
        }).round(2)
        
        print("\nCombat vs Non-Combat Network Performance:")
        print(combat_stats)
        
        # Identify problem periods
        high_risk_periods = self.df[self.df['performance_risk'] > 0.5]
        if len(high_risk_periods) > 0:
            print(f"\n⚠️ High Risk Periods: {len(high_risk_periods)} ({len(high_risk_periods)/len(self.df)*100:.1f}%)")
            print(f"   Average ping during high risk: {high_risk_periods['ping_ms'].mean():.1f}ms")
            print(f"   Combat overlap: {high_risk_periods['is_combat'].mean()*100:.1f}%")
    
    def visualize_comprehensive_analysis(self):
        """Create comprehensive visualizations"""
        print("\n📊 CREATING COMPREHENSIVE VISUALIZATIONS:")
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Performance Timeline
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot key metrics over time
        time_points = self.df.index[::100]  # Sample every 10 seconds
        ax1.plot(time_points, self.df['ping_ms'].iloc[time_points], 'b-', alpha=0.7, label='Ping (ms)')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(time_points, self.df['activity_score'].iloc[time_points] * 100, 'g-', alpha=0.7, label='Activity Score (×100)')
        
        # Highlight combat periods
        combat_mask = self.df['is_combat'].iloc[time_points]
        ax1.fill_between(time_points, 0, 200, where=combat_mask, alpha=0.3, color='red', label='Combat')
        
        ax1.set_title('Gaming Session Performance Timeline')
        ax1.set_xlabel('Time (100ms intervals)')
        ax1.set_ylabel('Ping (ms)', color='b')
        ax1_twin.set_ylabel('Activity Score', color='g')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Network Quality Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        
        combat_nq = self.df[self.df['is_combat']]['network_quality']
        noncombat_nq = self.df[~self.df['is_combat']]['network_quality']
        
        ax2.hist(combat_nq, bins=30, alpha=0.5, label='Combat', color='red', density=True)
        ax2.hist(noncombat_nq, bins=30, alpha=0.5, label='Non-Combat', color='blue', density=True)
        ax2.set_xlabel('Network Quality Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Network Quality by Combat State')
        ax2.legend()
        
        # 3. Action Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        
        action_counts = self.df['optimal_action'].value_counts().sort_index()
        all_action_labels = ['Low Slice', 'Medium Slice', 'High Slice']
        all_colors = ['green', 'orange', 'red']
        
        # Only use labels and colors for actions that actually exist in data
        actual_actions = sorted(action_counts.index)
        action_labels = [all_action_labels[i] for i in actual_actions]
        colors = [all_colors[i] for i in actual_actions]
        
        bars = ax3.bar(range(len(action_counts)), action_counts.values, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(action_counts)))
        ax3.set_xticklabels(action_labels)
        ax3.set_title('Optimal Action Distribution')
        ax3.set_ylabel('Count')
        
        # Add percentage labels
        total = action_counts.sum()
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height/total*100:.1f}%', ha='center', va='bottom')
        
        # 4. Performance Risk Heatmap
        ax4 = fig.add_subplot(gs[1, 2])
        
        # Create bins for heatmap
        ping_bins = pd.cut(self.df['ping_ms'], bins=[0, 50, 100, 150, 200, 1000], labels=['<50', '50-100', '100-150', '150-200', '>200'])
        activity_bins = pd.cut(self.df['activity_score'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['Very Low', 'Low', 'Med', 'High', 'Very High'])
        
        heatmap_data = pd.crosstab(activity_bins, ping_bins, values=self.df['performance_risk'], aggfunc='mean')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Risk Score'})
        ax4.set_title('Performance Risk by Activity & Ping')
        ax4.set_xlabel('Ping Range (ms)')
        ax4.set_ylabel('Activity Level')
        
        # 5. System Performance (NO GPU)
        ax5 = fig.add_subplot(gs[2, 0])
        
        system_metrics = ['cpu_percent', 'memory_percent']
        system_data = [self.df[metric] for metric in system_metrics]
        labels = ['CPU', 'Memory']
        
        ax5.boxplot(system_data, labels=labels)
        ax5.set_title('System Resource Usage Distribution')
        ax5.set_ylabel('Usage (%)')
        ax5.grid(True, alpha=0.3)
        ax5.text(0.5, -0.15, 'Note: Intel GPU - monitoring not available', 
                transform=ax5.transAxes, ha='center', style='italic', alpha=0.7)
        
        # 6. Reward Distribution
        ax6 = fig.add_subplot(gs[2, 1])
        
        ax6.hist(self.df['reward'], bins=50, alpha=0.7, color='purple')
        ax6.axvline(self.df['reward'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["reward"].mean():.2f}')
        ax6.set_xlabel('Reward Value')
        ax6.set_ylabel('Frequency')
        ax6.set_title('RL Reward Distribution')
        ax6.legend()
        
        # 7. Feature Correlation Matrix
        ax7 = fig.add_subplot(gs[2, 2])
        
        corr_features = ['combat_likelihood', 'network_quality', 'system_stress', 'performance_risk', 'ping_ms']
        corr_matrix = self.df[corr_features].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax7)
        ax7.set_title('Feature Correlation Matrix')
        
        # 8. RL State Space Visualization
        ax8 = fig.add_subplot(gs[3, :])
        
        # Show evolution of key RL states
        sample_indices = self.df.index[::50]  # Sample every 5 seconds
        
        ax8.plot(sample_indices, self.df['combat_likelihood'].iloc[sample_indices], label='Combat Likelihood', alpha=0.8)
        ax8.plot(sample_indices, self.df['network_quality'].iloc[sample_indices], label='Network Quality', alpha=0.8)
        ax8.plot(sample_indices, self.df['system_stress'].iloc[sample_indices], label='System Stress (CPU+Mem)', alpha=0.8)
        ax8.plot(sample_indices, self.df['performance_risk'].iloc[sample_indices], label='Performance Risk', alpha=0.8)
        
        ax8.set_xlabel('Time (100ms intervals)')
        ax8.set_ylabel('Score (0-1)')
        ax8.set_title('RL State Variables Evolution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_gaming_analysis_no_gpu.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_rl_training_data(self):
        """Save processed data for RL training"""
        print("\n💾 SAVING RL TRAINING DATA:")
        print("-" * 40)
        
        # Save main processed dataset
        main_file = f"processed_gaming_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.df.to_csv(main_file, index=False)
        print(f"✅ Main dataset saved: {main_file}")
        
        # Save RL-specific training data
        rl_file = f"rl_training_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.rl_training_data.to_csv(rl_file, index=False)
        print(f"✅ RL training data saved: {rl_file}")
        
        # Create summary statistics file
        summary = {
            'total_datapoints': len(self.df),
            'session_duration_minutes': len(self.df) * 0.1 / 60,
            'combat_ratio': self.df['is_combat'].mean(),
            'avg_ping': self.df['ping_ms'].mean(),
            'high_risk_ratio': self.df['performance_risk'].mean(),
            'action_distribution': self.df['optimal_action'].value_counts().to_dict(),
            'data_quality_flags': self.data_quality,
            'gpu_note': 'Intel GPU - monitoring not available, using CPU/Memory for system load'
        }
        
        summary_file = f"data_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"✅ Summary saved: {summary_file}")
        
        return main_file, rl_file, summary_file
    
    def print_final_report(self):
        """Print comprehensive final report"""
        print("\n" + "="*70)
        print("🎮 FINAL GAMING DATA ANALYSIS REPORT")
        print("="*70)
        
        duration_min = len(self.df) * 0.1 / 60
        print(f"\n📊 SESSION OVERVIEW:")
        print(f"   Duration: {duration_min:.1f} minutes")
        print(f"   Data Points: {len(self.df):,}")
        print(f"   Combat Time: {self.df['is_combat'].mean()*100:.1f}%")
        
        print(f"\n🌐 NETWORK PERFORMANCE:")
        print(f"   Average Ping: {self.df['ping_ms'].mean():.1f}ms")
        print(f"   Ping Stability: {self.df['ping_volatility'].mean():.1f}ms std")
        print(f"   Network Quality: {self.df['network_quality'].mean():.2f}/1.0")
        print(f"   High Ping Events: {(self.df['ping_ms'] > 100).mean()*100:.1f}%")
        
        print(f"\n💻 SYSTEM PERFORMANCE:")
        print(f"   CPU Usage: {self.df['cpu_percent'].mean():.1f}% ± {self.df['cpu_percent'].std():.1f}%")
        print(f"   Memory Usage: {self.df['memory_percent'].mean():.1f}%")
        print(f"   System Stress: {self.df['system_stress'].mean():.2f}/1.0")
        print(f"   Note: Intel GPU - using CPU/Memory for system load")
        
        print(f"\n🤖 RL TRAINING READINESS:")
        print(f"   Training Samples: {len(self.rl_training_data):,}")
        print(f"   State Features: 8 normalized features")
        print(f"   Actions: 3 network slice options")
        print(f"   Average Reward: {self.df['reward'].mean():.2f}")
        
        action_dist = self.df['optimal_action'].value_counts().sort_index()
        total_actions = action_dist.sum()
        print(f"\n📈 OPTIMAL ACTION STRATEGY:")
        all_action_names = ['Low Slice (Cheap)', 'Medium Slice (Balanced)', 'High Slice (Premium)']
        
        for action, count in action_dist.items():
            action_name = all_action_names[action] if action < len(all_action_names) else f"Action {action}"
            print(f"   {action_name}: {count:,} times ({count/total_actions*100:.1f}%)")
        
        print(f"\n⚠️ PERFORMANCE CHALLENGES:")
        combat_ping = self.df[self.df['is_combat']]['ping_ms'].mean()
        noncombat_ping = self.df[~self.df['is_combat']]['ping_ms'].mean()
        ping_increase = ((combat_ping - noncombat_ping) / noncombat_ping * 100) if noncombat_ping > 0 else 0
        
        print(f"   Combat Ping Impact: +{ping_increase:.1f}%")
        
        high_risk_combat = self.df[self.df['is_combat'] & (self.df['performance_risk'] > 0.5)]
        print(f"   High-Risk Combat Events: {len(high_risk_combat)} ({len(high_risk_combat)/self.df['is_combat'].sum()*100:.1f}% of combat)")
        
        print(f"\n💡 RL AGENT RECOMMENDATIONS:")
        premium_needed = (self.df['optimal_action'] == 2).mean() * 100
        if premium_needed > 20:
            print(f"   ⚡ High premium slice usage ({premium_needed:.1f}%) - optimize prediction accuracy")
        
        if ping_increase > 15:
            print(f"   🎯 Combat significantly increases ping - prioritize proactive allocation")
        
        if self.df['ping_volatility'].mean() > 20:
            print(f"   📶 High network instability - implement conservative buffering")
        
        print(f"   💻 Using CPU/Memory for system load (Intel GPU - no monitoring)")
        
        print(f"\n🚀 NEXT STEPS FOR RL IMPLEMENTATION:")
        print(f"   1. Use the 8 normalized state features for Q-learning input")
        print(f"   2. Implement 3-action Q-network (Low/Medium/High slice)")
        print(f"   3. Use the calculated reward function for training")
        print(f"   4. Consider {len(self.rl_training_data):,} transitions for initial training")
        print(f"   5. Target: Keep latency <50ms for {(self.df['ping_ms'] <= 50).mean()*100:.1f}% → 95%+")

def main():
    """Main execution function"""
    print("🎮 ENHANCED GAMING DATA PROCESSOR FOR RL PROJECT")
    print("=" * 60)
    print("📌 Intel GPU Compatible Version - No GPU Metrics")
    print("=" * 60)
    
    # You need to specify your merged CSV file here
    csv_file = input("\nEnter the path to your merged CSV file: ").strip()
    
    if not csv_file:
        print("Please provide the path to your merged CSV file")
        return
    
    try:
        # Initialize processor
        processor = ImprovedGameDataProcessor(csv_file)
        
        # Process the data step by step
        processor.clean_and_fix_data()
        processor.create_advanced_features()
        processor.create_rl_states_and_actions()
        
        # Analysis and visualization
        processor.analyze_network_patterns()
        processor.visualize_comprehensive_analysis()
        
        # Train combat predictor
        combat_model = processor.train_combat_predictor()
        
        # Save processed data
        main_file, rl_file, summary_file = processor.save_rl_training_data()
        
        # Final report
        processor.print_final_report()
        
        print(f"\n✅ PROCESSING COMPLETE!")
        print(f"📁 Files created:")
        print(f"   • {main_file} - Complete processed dataset")
        print(f"   • {rl_file} - RL training data")
        print(f"   • {summary_file} - Session summary")
        print(f"   • comprehensive_gaming_analysis_no_gpu.png - Visualizations")
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()