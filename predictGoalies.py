# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:57:03 2018

@author: Paul
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import (LinearRegression, BayesianRidge)
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             explained_variance_score,
                             r2_score)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from  matplotlib import pyplot as plt

path = 'C:/Users/Paul/Documents/ContentLab/9998-Hockey/'

min_hist_seasons = 0
tt_split = 'odd_'
cv = 2
features = 'agg_'

# get raw data
seasons = ['2010-11',
           '2011-12',
           '2012-13',
           '2013-14',
           '2014-15',
           '2015-16',
           '2016-17',
           '2017-18']
#seasons = ['2014-15']

columns = ['Last Name', 'First Name', 'Ht', 'Wt', 'GP', 'W', 'L', 'SA', 'SV%', 'GAA', 'MIN', 'SO']
df = pd.DataFrame()
for s in seasons:
    dfRaw = pd.read_excel(path + 'NHL_Goalies_' + s + '.xls')
    dfRaw = dfRaw.rename(columns = lambda x : x.strip())
    if (dfRaw.columns.isin(['HT']).any()):
        dfRaw['Ht'] = dfRaw['HT']
    dfS = pd.DataFrame(columns = columns)
    for c in columns:
        dfS[c] = dfRaw[c]
    dfS['Season'] = s
    df = df.append(dfS)
    
df['PTS'] = df.W + df.SO
   
# We need to keep string fields (Last Name, First Name, Season) as is,
# for joins and filtering across training and test data samples.
# The string field (Team) may be useful for the predictions,
# so encode this as a categorical variable.
#leTeam = LabelEncoder()
#df[['Team']] = df[['Team']].replace(np.nan, '').apply(leTeam.fit_transform)

if (features == 'agg_'):
    featureTargetSeasons = [(['2012-13', '2013-14'], ['2014-15']),
                            (['2015-16', '2016-17'], ['2017-18'])]
else:
    featureTargetSeasons = [(['2010-11'], ['2011-12']),
                            (['2012-13'], ['2013-14']),
                            (['2014-15'], ['2015-16']),
                            (['2016-17'], ['2017-18'])]

trainX = pd.DataFrame()
trainY = pd.Series()
testX = pd.DataFrame()
testY = pd.Series()
i = 0
while i < len(featureTargetSeasons):
    condFeature = (df.Season.isin(featureTargetSeasons[i][0]))
    c = df[condFeature].groupby(['Last Name', 'First Name']).count()
    featureData = df[condFeature].groupby(['Last Name', 'First Name']).agg(['mean', 'max'])
    featureData.columns = ["_".join(x) for x in featureData.columns.ravel()]
    featureData['numSeasons'] = c.Season
    
    condTarget = (df.Season.isin(featureTargetSeasons[i][1]))
    targetData = df[condTarget].groupby(['Last Name', 'First Name']).agg(['mean', 'max'])
    targetData.columns = ["_".join(x) for x in targetData.columns.ravel()]
    target = targetData.PTS_mean

    sampleData = featureData.join(target, how = 'inner', lsuffix = '_old', rsuffix = '_new').dropna()
    X = sampleData[sampleData.numSeasons > min_hist_seasons].drop(['PTS_mean_new'], axis = 1)
    Y = sampleData[sampleData.numSeasons > min_hist_seasons]['PTS_mean_new']
    
    split = 0
    if (tt_split == 'odd_'):
        split = 1
    if (i % 2 == split):
        trainX = trainX.append(X)
        trainY = trainY.append(Y)
    else:
        testX = testX.append(X)
        testY = testY.append(Y)
       
    i = i + 1

#regrTree = RandomForestRegressor(random_state=0, n_estimators=2000)
#regrTree = GradientBoostingRegressor(random_state=0, n_estimators=2000, loss = 'ls')
#regrTree = XGBRegressor(random_state=0, n_estimators=5000)
#regrLin = LinearRegression()
#regrLin = BayesianRidge()

#estimators = [('selectFeatures', PCA()),
#              ('regrTree', RandomForestRegressor(random_state=0))]
estimators = [('regrTree', RandomForestRegressor(random_state=0))]
pipe = Pipeline(estimators)
#paramGrid = dict(selectFeatures__n_components = [2, 5, 10],
#                 regrTree__n_estimators = [100, 1000, 10000])
#paramGrid = dict(regrTree__n_estimators = [1, 2, 5, 10, 15, 20, 25, 50, 100])
paramGrid = dict(regrTree__n_estimators = [100])
gridSearch = GridSearchCV(pipe, param_grid = paramGrid, cv = cv)
gridSearch.fit(trainX, trainY)
predictY = gridSearch.predict(testX)
mse = mean_squared_error(testY, predictY)
mae = mean_absolute_error(testY, predictY)
evs = explained_variance_score(testY, predictY)
r2 = r2_score(testY, predictY)
testResults = pd.DataFrame([[mse, mae, evs, r2]], columns = ['mse', 'mae', 'evs', 'r2'])
plt.figure(figsize = (8,8))
plt.scatter(testY, predictY)

fi = pd.DataFrame(columns = ['feature', 'importance'])
fi.feature = trainX.columns
fi.importance = gridSearch.best_estimator_.steps[0][1].feature_importances_
fis = fi.sort_values(by = 'importance')
x = np.arange(0,len(fis))
plt.figure(figsize = (8,12))
plt.barh(x, fis.importance)
plt.yticks(x, fis.feature)
plt.show()

filename = 'cvResults_goalies_' + features + tt_split + 'cv' + str(cv) + '.csv'
pd.DataFrame(gridSearch.estimator.steps).to_csv(path + filename, mode = 'a')
pd.DataFrame(gridSearch.cv_results_).to_csv(path + filename, mode = 'a')
testResults.to_csv(path + filename, mode = 'a')
fis.to_csv(path + filename, mode = 'a')
a = pd.DataFrame(testY, columns = ['test'])
a['predict'] = predictY
a.to_csv(path + filename, mode = 'a')
filename = 'dataset_goalies_' + features + tt_split + 'cv' + str(cv) + '.csv'
pd.DataFrame(featureTargetSeasons).to_csv(path + filename, mode = 'a')
trainX.head(5).to_csv(path + filename, mode = 'a')

# make predictions for 2018-19
condFeature = (df.Season.isin(['2017-18']))
c = df[condFeature].groupby(['Last Name', 'First Name']).count()
featureData = df[condFeature].groupby(['Last Name', 'First Name']).agg(['mean', 'max'])
featureData.columns = ["_".join(x) for x in featureData.columns.ravel()]
featureData['numSeasons'] = c.Season

poolPlayers = pd.read_csv(path + 'poolGoalies.csv')
targetData = poolPlayers.groupby(['Last Name', 'First Name']).agg(['mean', 'max'])
targetData.columns = ["_".join(x) for x in targetData.columns.ravel()]
target = targetData.PTS_mean

sampleData = featureData.join(target, how = 'inner', lsuffix = '_old', rsuffix = '_new').dropna()
X = sampleData[sampleData.numSeasons > min_hist_seasons].drop(['PTS_mean_new'], axis = 1)

predictions = gridSearch.predict(X)
results = poolPlayers
results['predictedPoints'] = predictions
plt.scatter(results.PTS, results.predictedPoints)   
results.to_csv(path + 'predictions_goalies_2018-19.csv')




    
    
    