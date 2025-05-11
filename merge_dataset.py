import pandas as pd

# Load your datasets
df1 = pd.read_csv("tcc_ceds_music.csv")  # or use pd.read_excel, pd.read_json, etc.
df2 = pd.read_csv("Student_dataset.csv")

# Merge datasets
merged_df = pd.concat([df1, df2], ignore_index=True)

# Optionally, remove duplicates if any
merged_df = merged_df.drop_duplicates()

# Save the merged dataset
merged_df.to_csv("merged_dataset.csv", index=False)
