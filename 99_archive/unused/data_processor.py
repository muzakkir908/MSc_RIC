import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import glob
import os

class EnhancedDataProcessor:
    def __init__(self, csv_file=None):
        # Load the most recent file if none specified
        if csv_file is None:
            files = glob.glob("../merged_enhanced_game_data*.csv")

            if files:
                csv_file = max(files, key=os.path.getctime)
                print(f"Loading: {csv_file}")
            else:
                print("No merged data files found! Run enhanced_game_collector.py first.")
                return
        
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} data points ({len(self.df) * 0.1 / 60:.1f} minutes)")
    
    def add_features(self):
        """Add derived features for better prediction"""
        # Network quality score
        self.df['network_quality'] = (
            (1 - self.df['ping_ms'] / 200) * 0.5 +  # Lower ping is better
            np.clip(self.df['bytes_recv_kbs'] / 1000, 0, 1) * 0.3 +  # Download speed
            np.clip(self.df['bytes_sent_kbs'] / 100, 0, 1) * 0.2  # Upload speed
        )
        
        # System load score
        self.df['system_load'] = (
            self.df['cpu_percent'] / 100 * 0.4 +
            self.df['memory_percent'] / 100 * 0.2 +
            self.df['gpu_percent'] / 100 * 0.4
        )
        
        # Lag risk score (combination of high ping and high system load)
        self.df['lag_risk'] = (
            (self.df['ping_ms'] > 80).astype(int) * 0.3 +
            (self.df['cpu_percent'] > 70).astype(int) * 0.3 +
            (self.df['gpu_percent'] > 80).astype(int) * 0.4
        )
        
        # Rolling averages for smoothing
        self.df['mouse_speed_avg'] = self.df['mouse_speed'].rolling(window=10).mean()
        self.df['ping_avg'] = self.df['ping_ms'].rolling(window=50).mean()
        
        # Fill NaN values
        # self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(0, inplace=True)
        
        print("✅ Enhanced features added!")
    
    def visualize_enhanced_data(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create a 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Network Performance Over Time
        ax1 = fig.add_subplot(gs[0, :2])
        ax1_twin = ax1.twinx()
        
        # Plot ping on primary axis
        ax1.plot(self.df.index, self.df['ping_ms'], 'b-', alpha=0.7, label='Ping (ms)')
        ax1.set_ylabel('Ping (ms)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot bandwidth on secondary axis
        ax1_twin.plot(self.df.index, self.df['bytes_recv_kbs'], 'g-', alpha=0.5, label='Download KB/s')
        ax1_twin.plot(self.df.index, self.df['bytes_sent_kbs'], 'r-', alpha=0.5, label='Upload KB/s')
        ax1_twin.set_ylabel('Bandwidth (KB/s)', color='g')
        
        # Highlight combat periods
        combat_periods = self.df[self.df['is_combat']]
        if len(combat_periods) > 0:
            for idx in combat_periods.index:
                ax1.axvspan(idx-5, idx+5, alpha=0.2, color='red')
        
        ax1.set_title('Network Performance During Gameplay')
        ax1.set_xlabel('Time (100ms intervals)')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. System Performance
        ax2 = fig.add_subplot(gs[0, 2])
        metrics = ['cpu_percent', 'memory_percent', 'gpu_percent']
        colors = ['blue', 'green', 'red']
        
        for metric, color in zip(metrics, colors):
            ax2.plot(self.df.index[::10], self.df[metric][::10], 
                    color=color, alpha=0.7, label=metric.replace('_percent', '').upper())
        
        ax2.set_title('System Resource Usage')
        ax2.set_ylabel('Usage %')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Combat vs Network Quality
        ax3 = fig.add_subplot(gs[1, 0])
        combat_df = self.df.groupby('is_combat').agg({
            'ping_ms': ['mean', 'std'],
            'network_quality': 'mean',
            'lag_risk': 'mean'
        }).round(2)
        
        x = ['Non-Combat', 'Combat']
        ping_means = [combat_df.loc[False, ('ping_ms', 'mean')], 
                     combat_df.loc[True, ('ping_ms', 'mean')]]
        ping_stds = [combat_df.loc[False, ('ping_ms', 'std')], 
                    combat_df.loc[True, ('ping_ms', 'std')]]
        
        ax3.bar(x, ping_means, yerr=ping_stds, capsize=10, color=['green', 'red'], alpha=0.7)
        ax3.set_title('Average Ping: Combat vs Non-Combat')
        ax3.set_ylabel('Ping (ms)')
        
        # Add value labels
        for i, v in enumerate(ping_means):
            ax3.text(i, v + 5, f'{v:.1f}ms', ha='center', va='bottom')
        
        # 4. Activity vs System Load Scatter
        ax4 = fig.add_subplot(gs[1, 1])
        combat_mask = self.df['is_combat']
        
        scatter = ax4.scatter(self.df['activity_score'], 
                            self.df['system_load'],
                            c=combat_mask, 
                            cmap='coolwarm', 
                            alpha=0.6,
                            s=30)
        
        ax4.set_xlabel('Activity Score')
        ax4.set_ylabel('System Load')
        ax4.set_title('Activity vs System Load (Red=Combat)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Combat State')
        
        # 5. Lag Risk Heatmap
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Create bins for heatmap
        ping_bins = pd.cut(self.df['ping_ms'], bins=[0, 50, 100, 150, 200, 1000])
        cpu_bins = pd.cut(self.df['cpu_percent'], bins=[0, 30, 50, 70, 90, 100])
        
        # Create pivot table for heatmap
        heatmap_data = pd.crosstab(cpu_bins, ping_bins, values=self.df['is_combat'], aggfunc='mean')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax5)
        ax5.set_title('Combat Probability by Ping & CPU')
        ax5.set_xlabel('Ping Range (ms)')
        ax5.set_ylabel('CPU Usage Range (%)')
        
        # 6. Network Quality Distribution
        ax6 = fig.add_subplot(gs[2, 0])
        
        ax6.hist(self.df[self.df['is_combat']]['network_quality'], 
                bins=30, alpha=0.5, label='Combat', color='red', density=True)
        ax6.hist(self.df[~self.df['is_combat']]['network_quality'], 
                bins=30, alpha=0.5, label='Non-Combat', color='blue', density=True)
        
        ax6.set_xlabel('Network Quality Score')
        ax6.set_ylabel('Density')
        ax6.set_title('Network Quality Distribution by Combat State')
        ax6.legend()
        
        # 7. Performance Timeline
        ax7 = fig.add_subplot(gs[2, 1:])
        
        # Create performance score
        perf_score = (
            (100 - self.df['ping_ms']) / 100 * 0.4 +
            (100 - self.df['cpu_percent']) / 100 * 0.3 +
            (100 - self.df['gpu_percent']) / 100 * 0.3
        )
        
        ax7.plot(self.df.index, perf_score, 'g-', alpha=0.7, label='Performance Score')
        
        # Mark combat periods
        # combat_starts = self.df[self.df['is_combat'] & ~self.df['is_combat'].shift(1)].index
        combat_starts = self.df[(self.df['is_combat'] == True) & (self.df['is_combat'].shift(1) == False)].index

        combat_ends = self.df[~self.df['is_combat'] & self.df['is_combat'].shift(1)].index
        
        for start, end in zip(combat_starts, combat_ends):
            ax7.axvspan(start, end, alpha=0.3, color='red', label='Combat' if start == combat_starts[0] else '')
        
        # Mark high lag risk periods
        high_risk = self.df[self.df['lag_risk'] > 0.5]
        if len(high_risk) > 0:
            ax7.scatter(high_risk.index, perf_score.iloc[high_risk.index], 
                       color='orange', s=20, alpha=0.8, label='High Lag Risk')
        
        ax7.set_xlabel('Time (100ms intervals)')
        ax7.set_ylabel('Performance Score')
        ax7.set_title('Overall Performance Timeline')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_game_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print enhanced statistics
        self.print_enhanced_stats()
    
    def print_enhanced_stats(self):
        """Print comprehensive statistics"""
        print("\n" + "="*60)
        print("📊 ENHANCED GAMING SESSION ANALYSIS")
        print("="*60)
        
        # Basic stats
        duration_min = len(self.df) * 0.1 / 60
        print(f"\n⏱️ Session Duration: {duration_min:.1f} minutes")
        print(f"📈 Total Data Points: {len(self.df):,}")
        
        # Combat statistics
        combat_ratio = self.df['is_combat'].mean() * 100
        combat_periods = self.df[self.df['is_combat'] != self.df['is_combat'].shift()].shape[0] // 2
        print(f"\n⚔️ COMBAT STATISTICS:")
        print(f"   Combat Time: {combat_ratio:.1f}%")
        print(f"   Combat Periods: {combat_periods}")
        print(f"   Avg Combat Duration: {self.df['is_combat'].sum() * 0.1 / combat_periods:.1f}s" if combat_periods > 0 else "   No combat detected")
        
        # Network performance
        print(f"\n🌐 NETWORK PERFORMANCE:")
        print(f"   Ping - Mean: {self.df['ping_ms'].mean():.1f}ms")
        print(f"   Ping - Min/Max: {self.df['ping_ms'].min():.1f}ms / {self.df['ping_ms'].max():.1f}ms")
        print(f"   Ping - Std Dev: {self.df['ping_ms'].std():.1f}ms")
        
        # Network during combat vs non-combat
        if combat_ratio > 0:
            combat_ping = self.df[self.df['is_combat']]['ping_ms'].mean()
            noncombat_ping = self.df[~self.df['is_combat']]['ping_ms'].mean()
            print(f"   Combat Ping: {combat_ping:.1f}ms vs Non-Combat: {noncombat_ping:.1f}ms")
            print(f"   Ping Increase During Combat: {((combat_ping/noncombat_ping - 1) * 100):.1f}%")
        
        print(f"   Download - Mean: {self.df['bytes_recv_kbs'].mean():.1f} KB/s")
        print(f"   Upload - Mean: {self.df['bytes_sent_kbs'].mean():.1f} KB/s")
        
        # System performance
        print(f"\n💻 SYSTEM PERFORMANCE:")
        print(f"   CPU - Mean: {self.df['cpu_percent'].mean():.1f}%")
        print(f"   CPU - Peak: {self.df['cpu_percent'].max():.1f}%")
        print(f"   Memory - Mean: {self.df['memory_percent'].mean():.1f}%")
        print(f"   GPU - Mean: {self.df['gpu_percent'].mean():.1f}%")
        print(f"   GPU - Peak: {self.df['gpu_percent'].max():.1f}%")
        
        # Performance issues
        print(f"\n⚠️ PERFORMANCE ISSUES:")
        high_ping_ratio = (self.df['ping_ms'] > 100).mean() * 100
        high_cpu_ratio = (self.df['cpu_percent'] > 80).mean() * 100
        high_gpu_ratio = (self.df['gpu_percent'] > 90).mean() * 100
        
        print(f"   High Ping (>100ms): {high_ping_ratio:.1f}% of time")
        print(f"   High CPU (>80%): {high_cpu_ratio:.1f}% of time")
        print(f"   High GPU (>90%): {high_gpu_ratio:.1f}% of time")
        
        # Lag risk analysis
        high_risk_ratio = (self.df['lag_risk'] > 0.5).mean() * 100
        print(f"   High Lag Risk: {high_risk_ratio:.1f}% of time")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS FOR RL AGENT:")
        if combat_ping > noncombat_ping * 1.2:
            print("   ⚡ Combat causes network strain - prioritize premium slices during combat")
        if high_cpu_ratio > 20:
            print("   🖥️ CPU bottleneck detected - consider edge processing")
        if self.df['ping_ms'].std() > 30:
            print("   📶 Network instability - implement adaptive buffering")
        if high_risk_ratio > 10:
            print("   🚨 Frequent lag risk - proactive slice allocation needed")
    
    def train_enhanced_predictor(self):
        """Train model with network and system features"""
        print("\n🤖 Training Enhanced Combat Predictor...")
        
        # Prepare features including network and system metrics
        feature_cols = [
            'mouse_speed', 'turning_rate', 'movement_keys', 'is_shooting',
            'activity_score', 'ping_ms', 'bytes_recv_kbs', 'bytes_sent_kbs',
            'cpu_percent', 'gpu_percent', 'network_quality', 'system_load'
        ]
        
        X = self.df[feature_cols]
        y = self.df['is_combat']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced combat/non-combat
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Model Performance:")
        print(f"   Accuracy: {accuracy:.2%}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Non-Combat', 'Combat'],
                                  digits=3))
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Feature Importance for Combat Prediction:")
        print("-" * 40)
        for _, row in importance_df.iterrows():
            bar_length = int(row['importance'] * 200)
            bar = '█' * bar_length
            print(f"{row['feature']:20} {row['importance']:.3f} {bar}")
        
        # Network impact analysis
        print("\n🌐 Network Impact on Combat Performance:")
        
        # Calculate correlation between network metrics and combat
        network_features = ['ping_ms', 'network_quality', 'bytes_recv_kbs']
        for feature in network_features:
            combat_mean = self.df[self.df['is_combat']][feature].mean()
            noncombat_mean = self.df[~self.df['is_combat']][feature].mean()
            diff_pct = ((combat_mean - noncombat_mean) / noncombat_mean * 100) if noncombat_mean != 0 else 0
            print(f"   {feature}: Combat {combat_mean:.2f} vs Non-Combat {noncombat_mean:.2f} ({diff_pct:+.1f}%)")
        
        return model
    
    def generate_rl_ready_data(self):
        """Prepare data for RL agent training"""
        print("\n🎮 Preparing Data for RL Agent...")
        
        # Create state representation for RL
        rl_data = pd.DataFrame()
        
        # Current state features
        rl_data['combat_probability'] = self.df['activity_score']  # Use as proxy
        rl_data['current_ping'] = self.df['ping_ms']
        rl_data['network_load'] = self.df['bytes_recv_kbs'] + self.df['bytes_sent_kbs']
        rl_data['system_load'] = self.df['system_load']
        
        # Create reward signal
        # Reward = good performance during combat, efficient resource use otherwise
        rl_data['reward'] = np.where(
            self.df['is_combat'],
            # Combat: reward low ping and good network
            (100 - self.df['ping_ms']) / 100 * 0.7 + self.df['network_quality'] * 0.3,
            # Non-combat: reward resource efficiency (lower is better)
            1 - (self.df['network_quality'] * 0.5 + self.df['system_load'] * 0.5)
        )
        
        # Action space (what the RL agent would choose)
        # 0: Normal slice, 1: Premium slice
        rl_data['optimal_action'] = np.where(
            (self.df['is_combat']) | (self.df['lag_risk'] > 0.5),
            1,  # Premium slice for combat or high lag risk
            0   # Normal slice otherwise
        )
        
        # Save RL-ready data
        rl_filename = f"rl_ready_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        rl_data.to_csv(rl_filename, index=False)
        
        print(f"✅ RL-ready data saved to {rl_filename}")
        print(f"   States: {len(rl_data)} time steps")
        print(f"   Premium slice needed: {(rl_data['optimal_action'] == 1).mean() * 100:.1f}% of time")
        
        return rl_data
    
    def save_processed_data(self):
        """Save the processed data with all features"""
        filename = f"processed_enhanced_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.df.to_csv(filename, index=False)
        print(f"\n💾 Processed data saved to {filename}")

if __name__ == "__main__":
    # Process the enhanced data
    processor = EnhancedDataProcessor()
    processor.add_features()
    processor.visualize_enhanced_data()
    model = processor.train_enhanced_predictor()
    rl_data = processor.generate_rl_ready_data()
    processor.save_processed_data()