import pandas as pd

# Load your CSV file
df = pd.read_csv('ska.csv')

# Convert 'release_date' to datetime and extract the year
df['release_date'] = pd.to_datetime(df['release_date'], format='%m/%d/%Y', errors='coerce').dt.year

# Save the modified CSV
df.to_csv('Student_dataset.csv', index=False)
