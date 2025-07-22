import pandas as pd

# Load both datasets
real_df = pd.read_csv('enhanced_game_data_20250603_194148.csv')  # Your 12k rows
synthetic_df = pd.read_csv('synthetic_game_data_20250604_000945.csv')  # 18k rows

# Merge and sort by timestamp
merged_df = pd.concat([real_df, synthetic_df], ignore_index=True)
merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)

# Save merged dataset
merged_df.to_csv('merged_enhanced_game_data.csv', index=False)
print(f"✅ Merged dataset: {len(merged_df)} rows total")