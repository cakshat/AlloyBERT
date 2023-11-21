import pandas as pd
from sklearn.model_selection import train_test_split


dataset = 'MPEA_clean'

# normalize YS, split into train and val sets
data = pd.read_csv(f'./data/{dataset}/{dataset}.csv')

if dataset == 'ys_clean':
    targetCol = 'YS'
else:
    targetCol = 'PROPERTY: Calculated Young modulus (GPa)'

data[targetCol] /= max(data[targetCol])
tr, vl = train_test_split(data, test_size=0.15)
tr.to_csv(f'./data/{dataset}/tr.csv')
vl.to_csv(f'./data/{dataset}/vl.csv')

# tr = pd.read_csv(f'./data/{dataset}/tr.csv')
ytr = tr[targetCol]
Xtr = tr.drop(targetCol, axis=1)

# vl = pd.read_csv(f'./data/{dataset}/vl.csv')
yvl = vl[targetCol]
Xvl = vl.drop(targetCol, axis=1)

texttr = []
for i, row in Xtr.iterrows():
    string = ''
    for k, v in row.items():
        string += f'{k}: {v}. '
    texttr.append(string)

textvl = []
for i, row in Xvl.iterrows():
    string = ''
    for k, v in row.items():
        string += f'{k}: {v}. '
    textvl.append(string)

df_train = pd.DataFrame()
df_train['text'] = texttr
df_train['target'] = ytr.tolist()

df_val = pd.DataFrame()
df_val['text'] = textvl
df_val['target'] = yvl.tolist()

df_train.to_pickle(f'./data/{dataset}/tr.pkl')
df_val.to_pickle(f'./data/{dataset}/vl.pkl')
