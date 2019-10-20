# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 08:51:27 2019

@author: mohammad ehtasham billah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import pickle
from sklearn.ensemble import AdaBoostClassifier


df = pd.read_excel("loan_data_final.xlsx")
df.columns
df.index = df["InstrumentID"]
df = df.drop(['InstrumentID', 'BOLink', 'BONote1', 
              'MunicipalityCode','TradingAreaDesc'], axis = 1)

df.head()
df.shape
df.dtypes
df.skew()
df.corr()
df.describe()

# creating dummy variables
df.dtypes
cat_features = ['has_swap_deal', 'SectorOfCounterparty', 'Typ', 'LegType',
       'AccountType', 'IFRS9', 'BucketALMM1', 'BucketNSFR',
       'BucketStoraExponeringar', 'PrincipalAmortType', 'CRD1Class2',
       'BucketRemainingMaturity', 'RiskWeightBucket']

df_final =pd.get_dummies(data = df, columns = cat_features,drop_first = True)


# splitting the data
x = df_final.drop("has_swap_deal_yes", axis = 1)
y = df_final["has_swap_deal_yes"]

test_size = 0.2
seed = 2019
x_train,x_test, y_train, y_test = train_test_split(x,y,
                                                   test_size = test_size,
                                                   random_state = seed)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_train_scaled = pd.DataFrame(x_train_scaled, columns = x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns = x_test.columns)


####################################################################
# fitting Gradient tree boosting
gbtr = GradientBoostingClassifier()

learning_rate_grid = [0.01,0.05,0.1]
n_estimator_grid = [100,200,300]
max_depth_grid = [2,3,5,7]
max_features_grid = ["auto", "log2"]

kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)

param_grid = dict(learning_rate = learning_rate_grid,
                  n_estimators = n_estimator_grid,
                  max_depth = max_depth_grid,
                  max_features = max_features_grid)

grid = GridSearchCV(estimator = gbtr, 
                    param_grid = param_grid, 
                    scoring = "roc_auc",
                    cv = kfold, 
                    verbose = 1)

grid_result = grid.fit(x_train_scaled,y_train)

print("Optimal parameters are {}".format(grid_result.best_params_))
print("Best score is {}.".format(grid_result.best_score_))

grid_result.cv_results_["mean_test_score"]  # why 72? for 72 possible grids?

# now fitting the model with optimal parameter
gbtr_opt = GradientBoostingClassifier(learning_rate = 0.05,
                                      max_depth = 7,
                                      max_features = "log2",
                                      n_estimators = 400,
                                      random_state = seed,
                                      verbose = 1
                                      )


gbtr_opt.fit(x_train_scaled,y_train)
gbtr_opt.train_score_

# variable importance
varimp_gbtr = pd.Series(gbtr_opt.feature_importances_, index = x_train_scaled.columns)
varimp_gbtr_sorted = varimp_gbtr.sort_values(axis=0,ascending = False,inplace = False, kind = "quicksort")
varimp_gbtr_sorted[:20]
varimp_gbtr_sorted[:20].plot(kind = "barh")
plt.show()

# prediction
y_hat_gbtr = gbtr_opt.predict(x_test_scaled)
y_hat_gbtr_prob = gbtr_opt.predict_proba(x_test_scaled)
print(classification_report(y_test,y_hat_gbtr))
cm_gbtr = confusion_matrix(y_test,y_hat_gbtr)
sns.heatmap(cm_gbtr,annot = True, fmt = "d")

# AUC ESTIMATION AND VISUALIZATION
auc_pred_gbtr = roc_auc_score(y_test, y_hat_gbtr_prob[:,1])
fpr, tpr, thresholds = roc_curve(y_test,y_hat_gbtr_prob[:,1])  # y_hat_gbtr_prob[:,1] means prob in favor of 1

print("AUC for final prediction is {} and cross validation is {}".format(round(auc_pred_gbtr,4),round(grid_result.best_score_,4)))
plt.plot(fpr,tpr, [0, 1], [0, 1],linestyle='--',marker = ".")

# save the model to disk
filename = 'gbtree_model.sav'
pickle.dump(gbtr_opt, open(filename, 'wb'))


# load the model from disk
"""
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

"""

##########################################################3

###########################################################
# adaboost classifer
adb = AdaBoostClassifier()

n_estimators_grid = [100,200,250,300,350,400,450,500]
learning_rate_grid = [.01,0.5,1.]

param_grid = dict(n_estimators = n_estimators_grid,
                  learning_rate = learning_rate_grid)

kfold= KFold(n_splits = 10, shuffle = True, random_state = seed)

grid_adb = GridSearchCV(estimator = adb,
                        param_grid = param_grid,
                        scoring = "roc_auc",
                        cv = kfold,
                        verbose = 1)

grid_result_adb =  grid_adb.fit(x_train_scaled,y_train)

print("Optimal parameters are {}".format(grid_result_adb.best_params_))
print("Best score is {}".format(grid_result_adb.best_score_))
grid_result_adb

# fitting the model with optimal parameters
adb_opt = AdaBoostClassifier(n_estimators = 400,
                             learning_rate = 0.5,
                             random_state = seed
                             )
adb_opt.fit(x_train_scaled,y_train)


# variable importance
varimp_adb = pd.Series(adb_opt.feature_importances_, index = x_train_scaled.columns)
varimp_adb_sorted = varimp_adb.sort_values(axis=0,ascending = False,inplace = False, kind = "quicksort")
varimp_adb_sorted[:20].plot(kind = "barh")
plt.show()


# prediction
yhat_adb = adb_opt.predict(x_test_scaled)
yhat_adb_prob = adb_opt.predict_proba(x_test_scaled)
print(classification_report(y_test,yhat_adb))
cm_adb = confusion_matrix(y_test,yhat_adb)
sns.heatmap(cm_adb, annot = True, fmt = "d")

# auc
auc_pred_adb = roc_auc_score(y_test, yhat_adb_prob[:,1])
fpr, tpr, thresholds = roc_curve(y_test,yhat_adb_prob[:,1])
print("AUC for final prediction is {} and cross validation is {}".format(round(auc_pred_adb,4),round(grid_result_adb.best_score_,4)))
plt.plot(fpr,tpr,[0, 1], [0, 1],linestyle='--',marker = ".")

# save the model to disk
filename = 'adb_model.sav'
pickle.dump(adb_opt, open(filename, 'wb'))


    

