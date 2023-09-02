"""
In SAS "Proc varclus " procedure used for variable clustering. It start with all variable putting in one cluster and then use PCA to split the data into predefined number of clusters
Python dont have equivalent package and function.

The OPTICS clustering algorithm are choosen due to its ease of implementation

The Silhouette Score is provided at overall level and for each variable. High score is good for clustering

modeler need to tune the min_features parameter this sets the minimum number of variable requires to cluster forms. smaller value creates lots of small clusters and large will create les cluster

Recommended Usage :
    1. First we want to remove very similar variables as we know only one can be used due to multi collinearity
    Things to consider:
    - largest IV/GINI/R2
    - easiest to interpret
    - most preferable data source
    Notes:
    - The sample parameter can be used when trailling min_features values, though make sure to run on full sample 
    - there is an option to automatically drop variables within the cluster and keep only the top 5 variables in it and  mean silhouette being greater than silhouette_threshold by default set to 0.5


"""

import joblib, json, sys, os, time, warnings, inspect,abc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)

# currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir=os.path.dirname(os.path.dirname(currentdir))

# homedir=os.path.dirname(os.path.dirname)

# sys.path.insert(0,parentdir)
# sys.path.insert(0,homedir)

#from credit_cards_era_model_development.src.variableclustering.variable_clustering1 import OPTICSVariableCluster
#from variable_clustering1 import OPTICSVariableCluster

class VariableCluster(abc.ABC):
    """Base Class for variable clustering"""

    def __init__(self, vars_to_cluster=None):
        """Constructor"""
        self._fitted=False
        self.vars_to_cluster=vars_to_cluster
        self.clusters=None
        self._mean_silhouette_score=None
        self.model=None
    @property
    def vars_to_cluster(self):
        return self._vars_to_cluster
    @vars_to_cluster.setter
    def clusters(self, value):
        if value is not None:
            if not isinstance(value,(pd.Index, list)):
                raise ValueError('vars_to_cluster must be type "list"')
            if isinstance(value, pd.Index):
                self._vars_to_cluster=value.tolist()
            else:
                self._vars_to_cluster=value
    @property
    def clusters(self):
        if not self._fitted:
            raise RuntimeError('Can not get cluster assignment before calling fit()')
        return self._clusters
    @clusters.setter
    def clusters(self,value):
        if value is not None:
            assert len(value) == len(self.vars_to_cluster)
        self._clusters=value
    def _check_columns(self,df):
        if df.shape[0]>1e5:
            warnings.warn('large number of samples may affect performance when clustering features consider down sampling',category=UserWarning)
        if not np.isin(self.vars_to_cluster, df.columns).all():
            missing_cols=[i for i in self.vars_to_cluster if i not in df.columns]
            raise ValueError(f"Dataframe must contain all variables in var_to_cluster. missing {missing_cols}.")
        return True
    @abc.abstractmethod
    def fit(self,df):
        """Fit the relevent clustering technique"""
    def mean_silhouette_score(self,df,metric='euclidean',**kwargs):
        if not self._fitted:
            raise RuntimeError('Must call fit() before scoring')
        self._check_columns(df)
        self._mean_silhouette_score=silhouette_score(df[self.vars_to_cluster].T, self.clusters, metric, **kwargs)
        return self._mean_silhouette_score
    def variable_silhouette_score(self, df, metric='euclidean', **kwargs):
        if not self._fitted:
            raise RuntimeError('Must call fit() before scoring')
        self._check_columns(df)
        self._silhouette_scores=silhouette_samples(df[self.vars_to_cluster].T, self.clusters, metric, **kwargs)
        return self._silhouette_scores
    def export_params(self, path=os.getcwd(),filename='VariableClusterFitted.pickle'):
        with open(os.path.join(path, filename),'wb') as file:
            pickle.dump(self,file)
    @staticmethod
    def load(path):
        """ Load a pickle file"""
        with open(path,'rb') as file:
            return pickle.load(file)
        
class KMeansVariableCluster(VariableCluster):
    def __init__(self, num_clusters, vars_to_cluster=None, n_init=10, random_state=None, n_iter=300, n_jobs=None, kwargs=None):
        """Constructor """
        super().__init__(vars_to_cluster)
        self.num_clusters=num_clusters
        self._n_init=n_init 
        self._random_state=random_state
        self._next_nearest_cluster=None
        self.centers=None
        self._n_iter=n_iter
        self._n_jobs=n_jobs 
        if kwargs:
            self._kwargs=kwargs
        else:
            self._kwargs={}
    @property
    def num_clusters(self):
        return self._num_clusters
    @num_clusters.setter
    def num_clusters(self, value):
        if not isinstance(value,int):
            raise ValueError('num clusters must be type "int"')
        elif value <=0:
            raise ValueError('Number of cluster must be greater or equal to 1. got {value}')
        self._num_clusters=value
    def fit(self,df):
        """Fit K means"""
        self._check_columns(df)
        self.model=KMeans(self.num_clusters, max_iter=self._n_iter, n_init=self._n_init, random_state=self._random_state, n_jobs=self._n_jobs, **self._kwargs)
        # Cluster on transpose for variable clustering
        self.model.fit(df[self.vars_to_cluster].T)
        self._fitted=True
        self.clusters=self.model.predict(df.T)
        self.centers=self.model.cluster_centers_
        return None
    def predict_new_labels(self,df):
        if not self._fitted:
            raise RuntimeError('Must call fit() before scoring')
        self._check_columns(df)
        return self.model.predict(df.T)
    def next_nearest_cluster(self,df):
        if not self._fitted:
            raise RuntimeError('Must call fit() before scoring')
        self._check_columns(df)
        # Get Second closest cluster details
        self._next_nearest_cluster=self.model.transform(df[self.vars_to_cluster].T).argpartition(1)[:,1]
        return self._next_nearest_cluster
class DBSCANVariableCluster(VariableCluster):
    def __init__(self, vars_to_cluster, eps=0.5, min_features=5, metric='euclidean',n_jobs=1, kwargs=None):
        """Constructor """
        super().__init__(vars_to_cluster)
        self.eps=eps
        self.min_features=min_features 
        self._metric=metric
        self._n_jobs=n_jobs 
        if kwargs:
            self._kwargs=kwargs
        else:
            self._kwargs={}
    @property
    def eps(self):
        return self._eps
    @eps.setter
    def eps(self, value):
        if not value > 0.0:
            raise ValueError(f" eps must be positive got value {value}")
        self._eps=value
    @property
    def min_features(self):
        return self._min_features
    @min_features.setter
    def min_features(self, value):
        if not isinstance(value,int):
            raise TypeError(f" min_features must be of type int got value {type(value)}")
        if not value >= 1:
            raise ValueError(f"min_feature must be greater than 0. Got {value}")
        self._min_features=value
    def fit(self,df):
        """ Fit DBSCAN for features of df"""
        self._check_columns(df)
        self.model=DBSCAN(self.eps,self.min_features,self._metric,)
