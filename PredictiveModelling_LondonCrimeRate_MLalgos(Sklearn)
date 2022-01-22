
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import sklearn
import sklearn.model_selection
from sklearn.model_selection import train_test_split
import sklearn.compose
import sklearn.preprocessing
import sklearn.svm
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import scipy.stats


# loading data
apply_df = pd.read_csv('/Users/yimin/project/export_LondonCrimeimd_Delta.csv')
# shape_apply_df =apply_df.shape
# test = apply_df.describe()

X = apply_df[["IncScore", "EmpScore", "EduScore", "HDDScore", "BHSScore", "EnvScore"]]
print("6 predictors")

#
# X = apply_df[["IncScore", "EmpScore", "EduScore", "HDDScore", "BHSScore", "EnvScore","CrimeRate_last_year",]]
# print("6 predictors with CrimeRate_last_year")

# X = apply_df[["IncScore", "EmpScore", "EduScore", "HDDScore", "BHSScore", "EnvScore", "NeiLevel"]]
# print("6 predictors with NeiLevel")

# X = apply_df[["IncScore", "EmpScore", "EduScore", "HDDScore", "BHSScore", "EnvScore", "NeiLevel", "CrimeRate_last_year"]]
# print("6 predictors with both two new features ")

Y = apply_df[["DeltaCrimeRate"]]


apply_df = pd.read_csv('/Users/yimin/project/export_LondonCrimeimd_Delta.csv')
shape_apply_df =apply_df.shape
X = apply_df[["IncScore", "EmpScore", "EduScore", "HDDScore", "BHSScore", "EnvScore", "CrimeRate_last_year", "NeiLevel"]]
Y = apply_df[["DeltaCrimeRate"]]

# # Y.hist()
# # plt.show()
# Y.hist(grid=False, rwidth=0.9, color='#0504aa')


# a naive regression model
normalizer_x = sklearn.preprocessing.StandardScaler()  # normalize x by z-score
normalizer_y = sklearn.preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
x_norm = normalizer_x.fit_transform(X.to_numpy())
y_norm = normalizer_y.fit_transform(Y.to_numpy()).reshape([len(Y)])
# kf = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
kf = sklearn.model_selection.KFold(n_splits=10, shuffle=False)
split_indexes = list(kf.split(x_norm))
pred = np.zeros_like(y_norm)
k = 0
for train_index, test_index in split_indexes:
    k += 1
    x_train, x_test = x_norm[train_index], x_norm[test_index]
    y_train, y_test = y_norm[train_index], y_norm[test_index]
    y_train_mean = np.average(y_train)
    pred[test_index] = y_train_mean

r2 = sklearn.metrics.r2_score(y_norm, pred)
r, _ = scipy.stats.pearsonr(y_norm, pred)
mse = sklearn.metrics.mean_squared_error(y_norm, pred)
mae = sklearn.metrics.mean_absolute_error(y_norm, pred)
print(f'Naive: R2={r2}, r ={r}, MSE={mse}, MAE={mae}')

def run(method='linear'):
    # X, Y = read_data()
    normalizer_x = sklearn.preprocessing.StandardScaler()  # normalize x by z-score
    normalizer_y = sklearn.preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
    x_norm = normalizer_x.fit_transform(X.to_numpy())
    y_norm = normalizer_y.fit_transform(Y.to_numpy()).reshape([len(Y)])
    # kf = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
    kf = sklearn.model_selection.KFold(n_splits=10, shuffle=False)
    split_indexes = list(kf.split(x_norm))
    pred = np.zeros_like(y_norm)
    feature_importances = []
    k = 0
    for train_index, test_index in split_indexes:
        k += 1
        x_train, x_test = x_norm[train_index], x_norm[test_index]
        y_train, y_test = y_norm[train_index], y_norm[test_index]

        if method == 'linear':
            regressor = sklearn.linear_model.LinearRegression()
        elif method == 'ridge':
            regressor = sklearn.linear_model.Ridge(alpha=0.1)
        elif method == 'lasso':
            regressor = sklearn.linear_model.Lasso(alpha=0.01)
        elif method == 'elasticNet':
            regressor = sklearn.linear_model.ElasticNet(alpha=0.01)
        elif method == 'svm':
            regressor = sklearn.svm.SVR(C=0.01)
        elif method == 'knn':
            regressor = sklearn.neighbors.KNeighborsRegressor(40)
        elif method == 'rf':
            # 6 predctors
            # regressor = sklearn.ensemble.RandomForestRegressor(max_features = "sqrt",min_samples_leaf=1000, random_state=1234)
            regressor = sklearn.ensemble.RandomForestRegressor(max_depth=10,max_features = "auto", random_state=1234)
        elif method == 'nn':
            regressor = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=100, learning_rate='adaptive',
                                                            early_stopping=True, n_iter_no_change=10, max_iter=500,
                                                            random_state=0)
        else:
            raise ValueError('Unknown method name: ', method)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    pred[test_index] = y_pred
    if method == 'rf':
        # get feature importance
        for tree in regressor.estimators_:
            feature_importances.append(tree.feature_importances_)

    r2 = sklearn.metrics.r2_score(y_norm, pred)
    r, _ = scipy.stats.pearsonr(y_norm, pred)
    mse = sklearn.metrics.mean_squared_error(y_norm, pred)
    mae = sklearn.metrics.mean_absolute_error(y_norm, pred)
    # print(f"{method}: R2={round(r2, 4)}, r = {round(adj_rsquared,4)} MSE={round(mse, 4)}, MAE={round(mae, 4)}")
    # print(f'{method}: R2={r2}, MSE={mse}')
    print(round(r2, 4),round(r,4),round(mse, 4),round(mae, 4))

    if method == 'rf':
        # compute mean and std of feature importance
        importances = np.array(feature_importances)
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        forest_importances = pd.Series(importances_mean, index=X.columns)
        # plot feature importances
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=importances_std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    run('linear')
    run('ridge')
    run('lasso')
    run('elasticNet')
    run('svm')
    run('knn')
    run('rf')
    run('nn')




