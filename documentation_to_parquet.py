import pandas as pd 

df = pd.read_excel('OldBailey/Documentation.xlsx')

# Set all columns to strings
df = df.astype(str)

# Set the FileID column as the index so we can search quicker
df = df.set_index("FileID").sort_index()

df.to_parquet("OldBailey/Documentation.parquet", engine='fastparquet')

