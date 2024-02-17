import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

df = pd.read_csv("rGME_dataset_features.csv", low_memory=False)

df = df[['date', 'compound']]

dates = sorted(set(df['date']))

TRAIN_START_DATE = '2021-01-04'
TRAIN_END_DATE = '2021-05-31'

# Compound value per date
vpd = []
for d in dates:
    x = df.loc[(df['date'] == d)]['compound']
    vpd.append([datetime.strptime(d, '%Y-%m-%d'), sum(x)/len(x)])

vpd = pd.DataFrame(vpd, columns=['date', 'compound'])
vpd.to_csv("date_compound.csv")

