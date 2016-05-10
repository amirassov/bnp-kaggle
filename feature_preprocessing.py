from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
import time
from tqdm import tqdm


class Factorize(BaseEstimator, TransformerMixin):

    def __init__(self, nan_strategy=-1):
        self.nan_strategy = nan_strategy
   
    def fit(self, X):
        self.indexers = []
        self.columns = X.columns
        for feature in tqdm(self.columns):
        	_, indexer = pd.factorize(X[feature])
        	self.indexers.append(indexer)
        return self
    
    def transform(self, X):
        facX = X.copy()
        for indexer, feature in zip(self.indexers, self.columns):
            facX[feature] = indexer.get_indexer(X[feature])
            facX.loc[facX[feature] == -1, feature] = self.nan_strategy
        return facX


class SuperImputer(BaseEstimator, TransformerMixin):

    def __init__(self, num_strategy='mean', 
    			cat_strategy='most_frequent',
    			num_features=[], cat_features=[]):
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.num_features = num_features
        self.cat_features = cat_features	

    def fit(self, X):
        if len(self.num_features) > 0:
            if isinstance(self.num_strategy, str):
                self.num_imp = Imputer(missing_values='NaN', strategy=self.num_strategy, axis=0)
                self.num_imp.fit(X[self.num_features])
        if len(self.cat_features) > 0:
            if isinstance(self.cat_strategy, str):
			self.cat_imp = Imputer(missing_values='NaN', strategy=self.cat_strategy, axis=0)
			self.cat_imp.fit(X[self.cat_features])
	
        return self
    
    def transform(self, X):
        if len(self.num_features) > 0:
            if isinstance(self.num_strategy, str):
                nimpX = pd.DataFrame(self.num_imp.transform(X[self.num_features]), 
	            									  columns=self.num_features)
            else:
	           nimpX = X[self.num_features].fillna(self.num_strategy)
        if len(self.cat_features) > 0:
            if isinstance(self.cat_strategy, str):
			cimpX = pd.DataFrame(self.cat_imp.transform(X[self.cat_features]), 
	            									   columns=self.cat_features)
            else:
                cimpX = X[self.cat_features].fillna(self.cat_strategy)
    	return pd.concat([nimpX, cimpX], axis=1)
