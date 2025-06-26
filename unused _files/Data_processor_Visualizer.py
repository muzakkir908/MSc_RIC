import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class GamingDataExplorer:
    def __init__(self, csv_file):
        """Initialize with your gaming CSV file"""
        self.df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(self.df):,} data points")
        print(f"⏱️  Session duration: {len(self.df) * 0.1 / 60:.1f} minutes")
        print(f"📅 Data collection rate: 10Hz (100ms intervals)")
        
    def explore_data_structure(self):
        """First look at the data structure"""
        print("\n🔍 DATA STRUCTURE:")
        print("-" * 50)
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print("\n📊 DATA TYPES:")
        print(self.df.dtypes.value_counts())
        
        print("\n❓ MISSING VALUES:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("✅ No missing values found!")
            
    def clean_data(self):
        """Basic data cleaning"""
        print("\n🧹 CLEANING DATA:")
        print("-" * 40)
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        print("✅ Converted timestamp to datetime")
        
        # Sort by timestamp
        self.df.sort_values('timestamp', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print("✅ Sorted data by timestamp")
        
        # Fill any missing values
        if self.df.isnull().sum().sum() > 0:
            self.df.fillna(method='ffill', inplace=True)
            self.df.fillna(0, inplace=True)
            print("✅ Filled missing values")
            
        # Add time-based features
        self.df['seconds_elapsed'] = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds()
        self.df['minutes_elapsed'] = self.df['seconds_elapsed'] / 60
        print("✅ Added time elapsed features")
        
    def analyze_gaming_metrics(self):
        """Analyze key gaming performance metrics"""
        print("\n🎮 GAMING METRICS ANALYSIS:")
        print("-" * 50)
        
        # Mouse activity
        print("\n🖱️  MOUSE ACTIVITY:")
        print(f"  Average mouse speed: {self.df['mouse_speed'].mean():.1f} pixels/s")
        print(f"  Max mouse speed: {self.df['mouse_speed'].max():.1f} pixels/s")
        print(f"  Average turning rate: {self.df['turning_rate'].mean():.1f} deg/s")
        print(f"  Static periods: {(self.df['mouse_speed'] == 0).mean()*100:.1f}%")
        
        # Combat analysis
        print("\n⚔️  COMBAT ANALYSIS:")
        combat_ratio = self.df['is_combat'].mean()
        print(f"  Combat time: {combat_ratio*100:.1f}%")
        print(f"  Total combat periods: {self.df['is_combat'].sum()}")
        print(f"  Shooting activity: {self.df['is_shooting'].mean()*100:.1f}%")
        
        # Network performance
        print("\n🌐 NETWORK PERFORMANCE:")
        print(f"  Average ping: {self.df['ping_ms'].mean():.1f}ms")
        print(f"  Ping range: {self.df['ping_ms'].min():.0f} - {self.df['ping_ms'].max():.0f}ms")
        print(f"  High ping (>100ms): {(self.df['ping_ms'] > 100).mean()*100:.1f}%")
        print(f"  Avg download: {self.df['bytes_recv_kbs'].mean():.1f} KB/s")
        print(f"  Avg upload: {self.df['bytes_sent_kbs'].mean():.1f} KB/s")
        
        # System performance
        print("\n💻 SYSTEM PERFORMANCE:")
        print(f"  CPU usage: {self.df['cpu_percent'].mean():.1f}%")
        print(f"  Memory usage: {self.df['memory_percent'].mean():.1f}%")
        
        # Check if GPU data exists
        if 'gpu_percent' in self.df.columns:
            gpu_mean = self.df['gpu_percent'].mean()
            if gpu_mean > 0:
                print(f"  GPU usage: {gpu_mean:.1f}%")
            else:
                print("  GPU: Intel GPU (no monitoring available)")
                
    def create_feature_correlations(self):
        """Calculate and display feature correlations"""
        print("\n🔗 FEATURE CORRELATIONS:")
        print("-" * 50)
        
        # Select key features for correlation
        key_features = [
            'ping_ms', 'cpu_percent', 'memory_percent',
            'mouse_speed', 'turning_rate', 'activity_score',
            'is_combat', 'is_shooting', 'bytes_recv_kbs'
        ]
        
        # Filter to existing columns
        available_features = [f for f in key_features if f in self.df.columns]
        
        # Calculate correlations
        corr_matrix = self.df[available_features].corr()
        
        # Find strong correlations
        print("\nStrong correlations (|r| > 0.3):")
        for i in range(len(available_features)):
            for j in range(i+1, len(available_features)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:
                    print(f"  {available_features[i]} ↔ {available_features[j]}: {corr_val:.3f}")
                    
    def visualize_comprehensive_analysis(self):
        """Create comprehensive visualizations"""
        print("\n📊 CREATING VISUALIZATIONS...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Performance Timeline
        ax1 = plt.subplot(4, 3, 1)
        sample_rate = 100  # Show every 10 seconds
        time_samples = self.df.index[::sample_rate]
        
        ax1.plot(self.df['minutes_elapsed'].iloc[time_samples], 
                self.df['ping_ms'].iloc[time_samples], 'b-', alpha=0.7, label='Ping (ms)')
        
        # Add combat highlighting
        combat_mask = self.df['is_combat'].iloc[time_samples]
        ax1.fill_between(self.df['minutes_elapsed'].iloc[time_samples], 
                        0, self.df['ping_ms'].iloc[time_samples].max(), 
                        where=combat_mask, alpha=0.2, color='red', label='Combat')
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Ping (ms)')
        ax1.set_title('Network Latency Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Activity Distribution
        ax2 = plt.subplot(4, 3, 2)
        ax2.hist(self.df['activity_score'], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(self.df['activity_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["activity_score"].mean():.2f}')
        ax2.set_xlabel('Activity Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Player Activity Distribution')
        ax2.legend()
        
        # 3. Combat vs Non-Combat Comparison
        ax3 = plt.subplot(4, 3, 3)
        combat_data = self.df[self.df['is_combat']]
        noncombat_data = self.df[~self.df['is_combat']]
        
        metrics = ['ping_ms', 'cpu_percent', 'mouse_speed']
        combat_means = [combat_data[m].mean() for m in metrics]
        noncombat_means = [noncombat_data[m].mean() for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x - width/2, combat_means, width, label='Combat', alpha=0.7, color='red')
        ax3.bar(x + width/2, noncombat_means, width, label='Non-Combat', alpha=0.7, color='blue')
        
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Average Value')
        ax3.set_title('Combat vs Non-Combat Performance')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Ping (ms)', 'CPU %', 'Mouse Speed'])
        ax3.legend()
        
        # 4. Mouse Movement Heatmap
        ax4 = plt.subplot(4, 3, 4)
        ax4.scatter(self.df['mouse_x'], self.df['mouse_y'], 
                   c=self.df['mouse_speed'], cmap='hot', alpha=0.5, s=1)
        ax4.set_xlabel('Mouse X Position')
        ax4.set_ylabel('Mouse Y Position')
        ax4.set_title('Mouse Movement Heatmap (colored by speed)')
        ax4.set_aspect('equal')
        
        # 5. System Resource Usage
        ax5 = plt.subplot(4, 3, 5)
        ax5.plot(self.df['minutes_elapsed'].iloc[time_samples], 
                self.df['cpu_percent'].iloc[time_samples], label='CPU %', alpha=0.7)
        ax5.plot(self.df['minutes_elapsed'].iloc[time_samples], 
                self.df['memory_percent'].iloc[time_samples], label='Memory %', alpha=0.7)
        
        ax5.set_xlabel('Time (minutes)')
        ax5.set_ylabel('Usage %')
        ax5.set_title('System Resource Usage Over Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Network Traffic
        ax6 = plt.subplot(4, 3, 6)
        ax6.plot(self.df['minutes_elapsed'].iloc[time_samples], 
                self.df['bytes_recv_kbs'].iloc[time_samples], 
                label='Download', alpha=0.7, color='green')
        ax6.plot(self.df['minutes_elapsed'].iloc[time_samples], 
                self.df['bytes_sent_kbs'].iloc[time_samples], 
                label='Upload', alpha=0.7, color='orange')
        
        ax6.set_xlabel('Time (minutes)')
        ax6.set_ylabel('KB/s')
        ax6.set_title('Network Traffic Over Time')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Ping Distribution by Combat State
        ax7 = plt.subplot(4, 3, 7)
        combat_ping = self.df[self.df['is_combat']]['ping_ms']
        noncombat_ping = self.df[~self.df['is_combat']]['ping_ms']
        
        ax7.hist(combat_ping, bins=30, alpha=0.5, label='Combat', color='red', density=True)
        ax7.hist(noncombat_ping, bins=30, alpha=0.5, label='Non-Combat', color='blue', density=True)
        ax7.set_xlabel('Ping (ms)')
        ax7.set_ylabel('Density')
        ax7.set_title('Ping Distribution by Combat State')
        ax7.legend()
        
        # 8. Key Press Activity
        ax8 = plt.subplot(4, 3, 8)
        key_metrics = ['movement_keys', 'action_keys', 'keys_pressed']
        key_data = [self.df[m].mean() for m in key_metrics if m in self.df.columns]
        key_labels = [m.replace('_', ' ').title() for m in key_metrics if m in self.df.columns]
        
        if key_data:
            ax8.bar(key_labels, key_data, alpha=0.7, color=['blue', 'green', 'orange'])
            ax8.set_ylabel('Average Keys Active')
            ax8.set_title('Keyboard Activity')
            ax8.set_xticklabels(key_labels, rotation=45)
        
        # 9. Feature Correlation Heatmap
        ax9 = plt.subplot(4, 3, 9)
        corr_features = ['ping_ms', 'cpu_percent', 'memory_percent', 
                        'mouse_speed', 'activity_score', 'is_combat']
        corr_features = [f for f in corr_features if f in self.df.columns]
        
        corr_matrix = self.df[corr_features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax9, square=True)
        ax9.set_title('Feature Correlation Matrix')
        
        # 10. Activity Score vs Ping
        ax10 = plt.subplot(4, 3, 10)
        scatter = ax10.scatter(self.df['activity_score'], self.df['ping_ms'], 
                              c=self.df['is_combat'], cmap='coolwarm', alpha=0.3, s=5)
        ax10.set_xlabel('Activity Score')
        ax10.set_ylabel('Ping (ms)')
        ax10.set_title('Activity vs Network Performance')
        plt.colorbar(scatter, ax=ax10, label='Combat')
        
        # 11. Time Series Decomposition
        ax11 = plt.subplot(4, 3, 11)
        # Calculate rolling averages
        window = 300  # 30 seconds
        self.df['ping_ma'] = self.df['ping_ms'].rolling(window=window, min_periods=1).mean()
        self.df['ping_std'] = self.df['ping_ms'].rolling(window=window, min_periods=1).std()
        
        ax11.plot(self.df['minutes_elapsed'].iloc[time_samples], 
                 self.df['ping_ma'].iloc[time_samples], 
                 label='30s Moving Average', linewidth=2)
        ax11.fill_between(self.df['minutes_elapsed'].iloc[time_samples],
                         (self.df['ping_ma'] - self.df['ping_std']).iloc[time_samples],
                         (self.df['ping_ma'] + self.df['ping_std']).iloc[time_samples],
                         alpha=0.3, label='±1 Std Dev')
        
        ax11.set_xlabel('Time (minutes)')
        ax11.set_ylabel('Ping (ms)')
        ax11.set_title('Network Stability Analysis')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        # 12. Summary Statistics Box
        ax12 = plt.subplot(4, 3, 12)
        ax12.axis('off')
        
        # Create summary text
        summary_text = f"""SESSION SUMMARY
        
Duration: {self.df['minutes_elapsed'].max():.1f} minutes
Total Samples: {len(self.df):,}

PERFORMANCE METRICS:
• Avg Ping: {self.df['ping_ms'].mean():.1f}ms
• Combat Time: {self.df['is_combat'].mean()*100:.1f}%
• Avg Activity: {self.df['activity_score'].mean():.2f}

SYSTEM USAGE:
• CPU: {self.df['cpu_percent'].mean():.1f}%
• Memory: {self.df['memory_percent'].mean():.1f}%

NETWORK STATS:
• Download: {self.df['bytes_recv_kbs'].mean():.1f} KB/s
• Upload: {self.df['bytes_sent_kbs'].mean():.1f} KB/s
• High Ping Events: {(self.df['ping_ms'] > 100).sum()}
"""
        
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('gaming_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved to 'gaming_data_analysis.png'")
        
    def generate_insights(self):
        """Generate key insights from the data"""
        print("\n💡 KEY INSIGHTS:")
        print("=" * 50)
        
        # Combat impact on performance
        combat_ping_avg = self.df[self.df['is_combat']]['ping_ms'].mean()
        noncombat_ping_avg = self.df[~self.df['is_combat']]['ping_ms'].mean()
        ping_increase = ((combat_ping_avg - noncombat_ping_avg) / noncombat_ping_avg * 100)
        
        print(f"\n1. COMBAT IMPACT:")
        print(f"   • Ping increases by {ping_increase:.1f}% during combat")
        print(f"   • Combat periods: {self.df['is_combat'].sum()} samples ({self.df['is_combat'].mean()*100:.1f}%)")
        
        # Network stability
        ping_std = self.df['ping_ms'].std()
        print(f"\n2. NETWORK STABILITY:")
        print(f"   • Ping stability: {ping_std:.1f}ms standard deviation")
        if ping_std > 30:
            print(f"   • ⚠️ High variability detected - unstable connection")
        else:
            print(f"   • ✅ Relatively stable connection")
            
        # Peak activity times
        high_activity = self.df[self.df['activity_score'] > self.df['activity_score'].quantile(0.75)]
        print(f"\n3. ACTIVITY PATTERNS:")
        print(f"   • High activity periods: {len(high_activity)} samples")
        print(f"   • Peak mouse speed: {self.df['mouse_speed'].max():.0f} pixels/s")
        
        # System bottlenecks
        print(f"\n4. SYSTEM PERFORMANCE:")
        if self.df['cpu_percent'].max() > 90:
            print(f"   • ⚠️ CPU bottleneck detected (max: {self.df['cpu_percent'].max():.1f}%)")
        if self.df['memory_percent'].max() > 85:
            print(f"   • ⚠️ Memory pressure detected (max: {self.df['memory_percent'].max():.1f}%)")
            
        # Network requirements
        print(f"\n5. NETWORK REQUIREMENTS:")
        print(f"   • Peak download: {self.df['bytes_recv_kbs'].max():.1f} KB/s")
        print(f"   • Peak upload: {self.df['bytes_sent_kbs'].max():.1f} KB/s")
        print(f"   • Recommended minimum bandwidth: {self.df['bytes_recv_kbs'].quantile(0.95)*8:.0f} Kbps")

def main():
    """Main execution function"""
    print("🎮 GAMING DATA EXPLORER")
    print("=" * 50)
    print("This tool will help you understand your gaming performance data\n")
    
    # Get file path
    csv_file = input("Enter the path to your gaming CSV file: ").strip()
    
    if not csv_file:
        print("❌ No file path provided!")
        return
        
    try:
        # Initialize explorer
        explorer = GamingDataExplorer(csv_file)
        
        # Run analysis steps
        explorer.explore_data_structure()
        explorer.clean_data()
        explorer.analyze_gaming_metrics()
        explorer.create_feature_correlations()
        explorer.visualize_comprehensive_analysis()
        explorer.generate_insights()
        
        print("\n✅ ANALYSIS COMPLETE!")
        print("Check 'gaming_data_analysis.png' for detailed visualizations")
        
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()