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
cv = 3
features = ''

# get raw data, then remove spike of players with points < 5, and flat area [5, 15],
# given that we are interested in predicting the points of high-points players
#df = pd.read_excel(path + 'NHL_2017-18.xls', header = 2)
dfRaw = pd.read_excel(path + 'NHL_1967-2018_plus2019.xls')
df = dfRaw.drop(['OGVT', 'DGVT', 'SGVT', 'GVT'], axis = 1) # these are not available after 2016
df = df[df.PTS > 15]

# We need to keep string fields (Last Name, First Name, Season) as is,
# for joins and filtering across training and test data samples.
# The string fields (Team, Pos) may be useful for the predictions,
# so encode these as categorical variables.
leTeam = LabelEncoder()
df[['Team']] = df[['Team']].replace(np.nan, '').apply(leTeam.fit_transform)
lePos = LabelEncoder()
df[['Pos']] = df[['Pos']].replace(np.nan, '').apply(lePos.fit_transform)

#featureTargetSeasons = [(['2002-03', '2003-04', '2004-05'], ['2005-06']),
#                        (['2006-07', '2007-08', '2008-09'], ['2009-10']),
#                        (['2010-11', '2011-12', '2012-13'], ['2013-14']),
#                        (['2014-15', '2015-16', '2016-17'], ['2017-18'])]
if (features == 'agg_'):
    featureTargetSeasons = [(['1967-68', '1968-69'], ['1969-70']),
                            (['1970-71', '1971-72'], ['1972-73']),
                            (['1973-74', '1974-75'], ['1975-76']),
                            (['1976-77', '1977-78'], ['1978-79']),
                            (['1979-80', '1980-81'], ['1981-82']),
                            (['1982-83', '1983-84'], ['1984-85']),
                            (['1985-86', '1986-87'], ['1987-88']),
                            (['1988-89', '1989-90'], ['1990-91']),
                            (['1991-92', '1992-93'], ['1993-94']),
                            (['1994-95', '1995-96'], ['1996-97']),
                            (['1997-98', '1998-99'], ['1999-00']),
                            (['2000-01', '2001-02'], ['2002-03']),
                            (['2003-04', '2004-05'], ['2005-06']),
                            (['2006-07', '2007-08'], ['2008-09']),
                            (['2009-10', '2010-11'], ['2011-12']),
                            (['2012-13', '2013-14'], ['2014-15']),
                            (['2015-16', '2016-17'], ['2017-18'])]
else:
    featureTargetSeasons = [(['1966-67'], ['1967-68']),
                            (['1968-69'], ['1969-70']),
                            (['1970-71'], ['1971-72']),
                            (['1972-73'], ['1973-74']),
                            (['1974-75'], ['1975-76']),
                            (['1976-77'], ['1977-78']),
                            (['1978-79'], ['1979-80']),
                            (['1980-81'], ['1981-82']),
                            (['1982-83'], ['1983-84']),
                            (['1984-85'], ['1985-86']),
                            (['1986-87'], ['1987-88']),
                            (['1988-89'], ['1989-90']),
                            (['1990-91'], ['1991-92']),
                            (['1992-93'], ['1993-94']),
                            (['1994-95'], ['1995-96']),
                            (['1996-97'], ['1997-98']),
                            (['1998-99'], ['1999-00']),
                            (['2000-01'], ['2001-02']),
                            (['2002-03'], ['2003-04']),
                            (['2004-05'], ['2005-06']),
                            (['2006-07'], ['2007-08']),
                            (['2008-09'], ['2009-10']),
                            (['2010-11'], ['2011-12']),
                            (['2012-13'], ['2013-14']),
                            (['2014-15'], ['2015-16']),
                            (['2016-17'], ['2017-18'])]
#featureTargetSeasons = [(['1998-99'], ['1999-00']),
#                        (['2000-01'], ['2001-02']),
#                        (['2002-03'], ['2003-04']),
#                        (['2004-05'], ['2005-06']),
#                        (['2006-07'], ['2007-08']),
#                        (['2008-09'], ['2009-10']),
#                        (['2010-11'], ['2011-12']),
#                        (['2012-13'], ['2013-14']),
#                        (['2014-15'], ['2015-16']),
#                        (['2016-17'], ['2017-18'])]

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
estimators = [('regrTree', XGBRegressor(random_state=0))]
pipe = Pipeline(estimators)
#paramGrid = dict(selectFeatures__n_components = [2, 5, 10],
#                 regrTree__n_estimators = [100, 1000, 10000])
#paramGrid = dict(regrTree__n_estimators = [5, 10, 15, 25, 50, 75, 100, 150, 200])
paramGrid = dict(regrTree__n_estimators = [50])
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

filename = 'cvResults_' + features + tt_split + 'cv' + str(cv) + '.csv'
pd.DataFrame(gridSearch.estimator.steps).to_csv(path + filename, mode = 'a')
pd.DataFrame(gridSearch.cv_results_).to_csv(path + filename, mode = 'a')
testResults.to_csv(path + filename, mode = 'a')
fis.to_csv(path + filename, mode = 'a')
a = pd.DataFrame(testY, columns = ['test'])
a['predict'] = predictY
a.to_csv(path + filename, mode = 'a')
filename = 'dataset_' + features + tt_split + 'cv' + str(cv) + '.csv'
pd.DataFrame(featureTargetSeasons).to_csv(path + filename, mode = 'a')
trainX.head(5).to_csv(path + filename, mode = 'a')

# make predictions for 2018-19
condFeature = (df.Season.isin(['2017-18']))
c = df[condFeature].groupby(['Last Name', 'First Name']).count()
featureData = df[condFeature].groupby(['Last Name', 'First Name']).agg(['mean', 'max'])
featureData.columns = ["_".join(x) for x in featureData.columns.ravel()]
featureData['numSeasons'] = c.Season

poolPlayers = pd.read_csv(path + 'poolPlayers.csv')
targetData = poolPlayers.groupby(['Last Name', 'First Name']).agg(['mean', 'max'])
targetData.columns = ["_".join(x) for x in targetData.columns.ravel()]
target = targetData.PTS_mean

sampleData = featureData.join(target, how = 'inner', lsuffix = '_old', rsuffix = '_new').dropna()
X = sampleData[sampleData.numSeasons > min_hist_seasons].drop(['PTS_mean_new'], axis = 1)

predictions = gridSearch.predict(X)
results = poolPlayers
results['predictedPoints'] = predictions
plt.scatter(results.PTS, results.predictedPoints)
results.to_csv(path + 'predictions_2018-19.csv')





    
    
    