import pandas as pd
import random
p = 0.012  # 1% of the lines
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
random.seed(0)
filename = '~/Downloads/train.csv'
df = pd.read_csv(
         filename,
         header=0, 
         skiprows=lambda i: i>0 and random.random() > p
)
cols = list(df.columns)
cols = [cols[-1]] + cols[:-1]
df = df[cols]

print(df.iloc[0])

df.to_csv('train_cifar.csv', index=False)