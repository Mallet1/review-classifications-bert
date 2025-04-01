import pandas as pd

df = pd.read_csv('src/data/labeled_data.csv')

print(df['sentiment'].value_counts())

