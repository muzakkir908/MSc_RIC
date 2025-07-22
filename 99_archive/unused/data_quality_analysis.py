import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_data_quality(data_path):
    """Comprehensive data quality analysis"""
    print("🔍 DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} data points\n")
    
    # 1. Class Distribution Analysis
    print("1. CLASS DISTRIBUTION:")
    print("-" * 30)
    combat_ratio = df['is_combat'].mean()
    print(f"Combat events: {df['is_combat'].sum():,} ({combat_ratio:.1%})")
    print(f"Non-combat events: {(~df['is_combat']).sum():,} ({1-combat_ratio:.1%})")
    
    # Check combat sequence lengths
    combat_changes = df['is_combat'].diff().fillna(0).abs()
    combat_segments = (combat_changes.cumsum() + 1) // 2
    
    combat_lengths = []
    for segment in df.groupby(combat_segments):
        if segment[1]['is_combat'].iloc[0]:
            combat_lengths.append(len(segment[1]))
    
    if combat_lengths:
        print(f"\nCombat sequence statistics:")
        print(f"  Average length: {np.mean(combat_lengths):.1f} timesteps ({np.mean(combat_lengths)*0.1:.1f}s)")
        print(f"  Min length: {min(combat_lengths)} timesteps")
        print(f"  Max length: {max(combat_lengths)} timesteps")
    
    # 2. Feature Quality Analysis
    print("\n2. FEATURE QUALITY:")
    print("-" * 30)
    
    key_features = ['mouse_speed', 'turning_rate', 'is_shooting', 'movement_keys', 'activity_score']
    existing_features = [f for f in key_features if f in df.columns]
    
    for feature in existing_features:
        combat_mean = df[df['is_combat']][feature].mean()
        noncombat_mean = df[~df['is_combat']][feature].mean()
        
        # T-test for difference
        t_stat, p_value = stats.ttest_ind(
            df[df['is_combat']][feature].dropna(),
            df[~df['is_combat']][feature].dropna()
        )
        
        print(f"\n{feature}:")
        print(f"  Combat avg: {combat_mean:.2f}")
        print(f"  Non-combat avg: {noncombat_mean:.2f}")
        print(f"  Difference: {abs(combat_mean - noncombat_mean):.2f}")
        print(f"  Statistical significance: {'YES' if p_value < 0.05 else 'NO'} (p={p_value:.4f})")
    
    # 3. Data Consistency Checks
    print("\n3. DATA CONSISTENCY:")
    print("-" * 30)
    
    # Check for suspicious patterns
    static_mouse = ((df['mouse_speed'] == 0) & (df['turning_rate'] == 0)).sum()
    print(f"Static mouse periods: {static_mouse} ({static_mouse/len(df)*100:.1f}%)")
    
    # Check shooting without combat
    shooting_noncombat = (df['is_shooting'] & ~df['is_combat']).sum()
    print(f"Shooting in non-combat: {shooting_noncombat} ({shooting_noncombat/df['is_shooting'].sum()*100:.1f}%)")
    
    # Check if combat labels might be delayed
    print("\nChecking for label delays...")
    correlations = []
    for shift in range(-50, 51, 10):  # -5s to +5s
        shifted_combat = df['is_combat'].shift(shift).fillna(0)
        corr = df['is_shooting'].corr(shifted_combat)
        correlations.append((shift, corr))
    
    best_shift = max(correlations, key=lambda x: x[1])
    print(f"Best correlation between shooting and combat: {best_shift[1]:.3f} at {best_shift[0]*0.1:.1f}s shift")
    
    # 4. Visualization
    print("\n4. CREATING VISUALIZATIONS...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Combat distribution over time
    ax = axes[0, 0]
    combat_rolling = df['is_combat'].rolling(1000).mean()
    ax.plot(combat_rolling)
    ax.set_title('Combat Ratio Over Time (Rolling Avg)')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Combat Ratio')
    
    # Feature distributions by class
    ax = axes[0, 1]
    feature = 'mouse_speed' if 'mouse_speed' in df.columns else existing_features[0]
    df[df['is_combat']][feature].hist(bins=30, alpha=0.5, label='Combat', ax=ax, density=True)
    df[~df['is_combat']][feature].hist(bins=30, alpha=0.5, label='Non-Combat', ax=ax, density=True)
    ax.set_title(f'{feature} Distribution by Class')
    ax.legend()
    
    # Activity score vs combat
    ax = axes[0, 2]
    if 'activity_score' in df.columns:
        sample = df.sample(min(5000, len(df)))
        ax.scatter(sample['activity_score'], sample['is_combat'], alpha=0.1)
        ax.set_xlabel('Activity Score')
        ax.set_ylabel('Is Combat')
        ax.set_title('Activity Score vs Combat Label')
    
    # Correlation heatmap
    ax = axes[1, 0]
    corr_features = [f for f in existing_features if df[f].dtype in ['float64', 'int64']]
    if len(corr_features) > 1:
        corr_matrix = df[corr_features + ['is_combat']].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlations')
    
    # Combat sequence lengths
    ax = axes[1, 1]
    if combat_lengths:
        ax.hist(combat_lengths, bins=30, edgecolor='black')
        ax.set_xlabel('Sequence Length (timesteps)')
        ax.set_ylabel('Count')
        ax.set_title('Combat Sequence Length Distribution')
        ax.axvline(30, color='red', linestyle='--', label='30 timesteps')
        ax.legend()
    
    # Missing data
    ax = axes[1, 2]
    missing = df[existing_features].isnull().sum()
    if missing.sum() > 0:
        missing.plot(kind='bar', ax=ax)
        ax.set_title('Missing Values by Feature')
        ax.set_ylabel('Count')
    else:
        ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Data Completeness Check')
    
    plt.tight_layout()
    plt.savefig('data_quality_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 5. Recommendations
    print("\n5. RECOMMENDATIONS:")
    print("-" * 30)
    
    if combat_ratio < 0.2:
        print("⚠️ Very low combat ratio - consider:")
        print("   - Collecting more combat-heavy gameplay sessions")
        print("   - Using data augmentation techniques")
        print("   - Adjusting prediction window (currently 2.5s ahead)")
    
    if best_shift[0] != 0:
        print(f"⚠️ Possible label misalignment detected ({best_shift[0]*0.1:.1f}s)")
        print("   - Consider adjusting label timing in preprocessing")
    
    if static_mouse > len(df) * 0.3:
        print("⚠️ High percentage of static periods")
        print("   - May indicate idle time or data collection issues")
        print("   - Consider filtering out long idle periods")
    
    print("\n✅ Analysis complete! Check 'data_quality_analysis.png' for visualizations.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_data_quality(sys.argv[1])
    else:
        data_path = input("Enter path to processed gaming data CSV: ").strip()
        if data_path:
            analyze_data_quality(data_path)
        else:
            print("No data path provided.")