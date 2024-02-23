import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt


dataset = 'MPEA_numeric'

tr = pd.read_csv(f'./data/{dataset}/tr.csv')
vl = pd.read_csv(f'./data/{dataset}/vl.csv')

if dataset == 'ys_clean':
    targetCol = 'YS'
else:
    targetCol = 'PROPERTY: Calculated Young modulus (GPa)'

ytr = tr[targetCol]
Xtr = tr.drop(targetCol, axis=1)

yvl = vl[targetCol]
Xvl = vl.drop(targetCol, axis=1)


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
rf = RandomForestRegressor(max_depth=10, random_state=0).fit(Xtr, ytr)
yh2 = rf.predict(Xvl)
RF_loss = np.mean((yh2 - yvl)**2)
print('Random Forest Model:', RF_loss)
plt.clf()
plt.plot(yh2)
plt.plot(yvl.to_numpy())
plt.savefig(f'./graphs/RF/{dataset}.png')


# SVR
svr = SVR(C=0.1).fit(Xtr, ytr)
yh3 = svr.predict(Xvl)
SVR_loss = np.mean((yh3 - yvl)**2)
print('SVR Model:', SVR_loss)
plt.clf()
plt.plot(yh3)
plt.plot(yvl.to_numpy())
plt.savefig(f'./graphs/SVR/{dataset}.png')


# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, max_depth=20).fit(Xtr, ytr)
yh4 = gb.predict(Xvl)
GB_loss = np.mean((yh4 - yvl)**2)
print('Gradient Boosting Model:', GB_loss)
plt.clf()
plt.plot(yh4)
plt.plot(yvl.to_numpy())
plt.savefig(f'./graphs/GB/{dataset}.png')
