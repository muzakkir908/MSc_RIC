import pandas as pd
import os

# Define all 3 full file paths
files_to_merge = [
    r"C:\Users\muzak\OneDrive\Desktop\NCI\MSc Project\cloud_gaming_project\01_data\02_processed\processed_gaming_data_20250604_130015.csv",
    r"C:\Users\muzak\OneDrive\Desktop\NCI\MSc Project\cloud_gaming_project\01_data\02_processed\processed_gaming_data_20250605_103542.csv",
    r"C:\Users\muzak\OneDrive\Desktop\NCI\MSc Project\cloud_gaming_project\01_data\02_processed\processed_gaming_data_20250730_135540.csv"
]

# Load all CSVs
dfs = [pd.read_csv(f) for f in files_to_merge]

# Combine all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Remove duplicates based on timestamp
merged_df = merged_df.drop_duplicates(subset='timestamp')

# Sort by timestamp
merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)

# Output path
output_path = r"C:\Users\muzak\OneDrive\Desktop\NCI\MSc Project\cloud_gaming_project\01_data\02_processed\merged_processed_gaming_data.csv"

# Save the result
merged_df.to_csv(output_path, index=False)

# Print summary
print(f"✅ Merged {len(files_to_merge)} files into:\n{output_path}")
print(f"📊 Final rows after removing timestamp duplicates: {len(merged_df)}")
