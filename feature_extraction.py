from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from tqdm import tqdm


class NanEncoding(BaseEstimator, TransformerMixin):

    def fit(self, X):
        nanX = X.isnull()
        self.features = X.columns[nanX.sum(axis=0) > 0]
        return self

    def transform(self, X):
        nanX = X.isnull()
        return nanX[self.features].astype(int)

class Rank(BaseEstimator, TransformerMixin):

    def fit(self, X):
        strX = X.astype(str)
        self.value_series = []
        for f in tqdm(strX.columns):
            sort_values = np.sort(np.array(list(set(strX[f]))))
            self.value_series.append(pd.Series(index=sort_values, data=np.arange(len(sort_values))))
        return self

    def transform(self, X):
        rankX = pd.DataFrame()
        strX = X.astype(str)
        for i, f in tqdm(enumerate(strX.columns)):
            rankX[f] = np.array(self.value_series[i][strX[f].values])
            rankX.loc[X[f].isnull().values, f] = np.nan
        return rankX


class RankCount(BaseEstimator, TransformerMixin):

    def fit(self, X):
        strX = X.astype(str)
        self.value_series = []
        for f in tqdm(strX.columns):
            sort_values = strX[f].value_counts().index.values
            self.value_series.append(pd.Series(index=sort_values, data=np.arange(len(sort_values))))
        return self

    def transform(self, X):
        rankX = pd.DataFrame()
        strX = X.astype(str)
        for i, f in tqdm(enumerate(strX.columns)):
            rankX[f] = np.array(self.value_series[i][strX[f].values])
            rankX.loc[X[f].isnull().values, f] = np.nan
        return rankX


class CountEncoding():

    def __init__(self, n_folds=10, shuffle=True, random_state=44, verbose=False):
        self.verbose = verbose
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.sum_full_counts = []
        self.sum_pos_counts = []

    def fit_transform(self, X, y):
        X_full_count = pd.DataFrame(index=X.index)
        X_pos_count = pd.DataFrame(index=X.index)
        X_smooth_count = pd.DataFrame(index=X.index)
        cv = StratifiedKFold(y, n_folds=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)
        for i, feature in tqdm(enumerate(X.columns)):
            self.sum_full_counts.append(X[feature].value_counts() * 0.0)
            self.sum_pos_counts.append(X[feature][y == 1].value_counts() * 0.0)

            for train, test in cv:
                full_counts = X.ix[train][feature].value_counts().astype(float)
                pos_counts = X[y == 1].ix[train][feature].value_counts().astype(float)

                self.sum_full_counts[i][full_counts.index] += full_counts
                self.sum_pos_counts[i][pos_counts.index] += pos_counts

                X_full_count.ix[test, feature] = np.array(full_counts[X.ix[test][feature]])
                X_pos_count.ix[test, feature]  = np.array(pos_counts[X.ix[test][feature]])

            self.sum_full_counts[i] /= self.n_folds
            self.sum_pos_counts[i] /= self.n_folds

            X_smooth_count[feature] = (X_pos_count[feature] + 1.0) / (X_full_count[feature] + 2.0)

        return X_smooth_count.fillna(0)

    def transform(self, X):
        X_full_count = pd.DataFrame(index=X.index)
        X_pos_count = pd.DataFrame(index=X.index)
        X_smooth_count = pd.DataFrame(index=X.index)

        for i, feature in tqdm(enumerate(X.columns)):
            X_full_count[feature] = np.array(self.sum_full_counts[i][X[feature]])
            X_pos_count[feature] = np.array(self.sum_pos_counts[i][X[feature]])
            X_smooth_count[feature] = (X_pos_count[feature] + 1.0) / (X_full_count[feature] + 2.0)
        return X_smooth_count.fillna(0)


def make_pair_features(df_train, df_test):
    pair_train = pd.DataFrame()
    pair_test = pd.DataFrame()
    for i, f1 in tqdm(enumerate(df_train.columns)):
        for j, f2 in enumerate(df_train.columns):
            if i < j:
                pair_train[f1 + f2] = df_train[f1] + '-' + df_train[f2]
                pair_test[f1 + f2] = df_test[f1] + '-' + df_test[f2]
    return pair_train, pair_test


class SuperCatEncoder():

    def __init__(self,
                 strategy=['mean', 'min', 'max'],
                 cat_features=[],
                 num_features=[]):
        self.strategy = strategy
        self.cat_features = cat_features
        self.num_features = num_features

    def get(self, X):
        encX = pd.DataFrame(index=X.index)
        for cat_feature in tqdm(self.cat_features):
            unique_values = set(X[cat_feature])
            for value in unique_values:
                numX = X[self.num_features][X[cat_feature] == value]
                for num_feature in self.num_features:
                    if 'mean' in self.strategy:
                        encX.ix[numX.index, cat_feature + num_feature + '_mean'] = numX[num_feature].mean(axis=0)
                    if 'max' in self.strategy:
                        encX.ix[numX.index, cat_feature + num_feature + '_max'] = numX[num_feature].max(axis=0)
                    if 'min' in self.strategy:
                        encX.ix[numX.index, cat_feature + num_feature + '_min'] = numX[num_feature].min(axis=0)
        return encX
