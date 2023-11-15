import pandas as pd
from sklearn.model_selection import train_test_split


# normalize YS, split into train and val sets
data = pd.read_csv('./data/ys_clean.csv')
data['YS'] /= max(data['YS'])
tr, vl = train_test_split(data, test_size=0.15)
tr.to_csv('./data/ys_clean_tr.csv')
vl.to_csv('./data/ys_clean_vl.csv')

# tr = pd.read_csv('./data/ys_clean_tr.csv')
ytr = tr['YS']
Xtr = tr.drop('YS', axis=1)

# vl = pd.read_csv('./data/ys_clean_vl.csv')
yvl = vl['YS']
Xvl = vl.drop('YS', axis=1)

targettr = ytr.tolist()
targetvl = yvl.tolist()

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
df_train['target'] = targettr

df_val = pd.DataFrame()
df_val['text'] = textvl
df_val['target'] = targetvl

df_train.to_pickle('./data/train.pkl')
df_val.to_pickle('./data/val.pkl')
