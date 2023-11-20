import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


dataset = 'ys_clean'

tr = pd.read_csv(f'./data/{dataset}_tr.csv')
vl = pd.read_csv(f'./data/{dataset}_vl.csv')

ytr = tr['YS']
Xtr = tr.drop('YS', axis=1)

yvl = vl['YS']
Xvl = vl.drop('YS', axis=1)


# Linear Regression
reg = LinearRegression().fit(Xtr, ytr)
yh = reg.predict(Xvl)
LR_loss = np.mean((yh - yvl)**2)
print('Linear Regression Model:', LR_loss)
plt.figure(figsize=(18, 5))
plt.plot(yh)
plt.plot(yvl.to_numpy())
plt.savefig(f'./graphs/LR/{dataset}.png')


# Random Forest
rf = RandomForestRegressor(max_depth=2, random_state=0).fit(Xtr, ytr)
yh2 = rf.predict(Xvl)
RF_loss = np.mean((yh2 - yvl)**2)
print('Random Forest Model:', RF_loss)
plt.clf()
plt.plot(yh2)
plt.plot(yvl.to_numpy())
plt.savefig(f'./graphs/RF/{dataset}.png')
