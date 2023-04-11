import pandas as pd

df = pd.read_csv('memray-csv-dp_time_space_test.py.54347.csv')
# Dropping the team 1
df = df[df["stack_trace"].str.contains("python3.8/site-packages/dataprofiler") == True]
total_dp_size_alloc = round(sum(df["size"])/1e9, 3)
print(f"{total_dp_size_alloc} GB")