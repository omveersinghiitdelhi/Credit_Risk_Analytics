""" Predictive Power"""

import sys, os,joblib
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pandas.api.types import is_object_dtype
from tqdm import tqdm
import statsmodels.api as sm

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.functions.transformations.transformation_functions import *

def iv(df, var, target_variable, weight=None, bins=20):
    """Function to find IV of a column (numeric or categorical)"""
    _df=df.copy(deep=True)
    if set(_df[target_variable].unique())=={0,1}:
        if weight is None:
            data=df[[target_variable,var]].copy()
            data.loc[:,'wgt']=1
        else:
            df['wgt']=df[weight]
            data=df[[target_variable, var, 'wgt']].copy()
        
        if np.issubdtype(data[var].dtype,np.number):
            n_unique=data[var].nunique() # Accounting for missing values
            data.loc[:,'rank']=data[var].rank(method='dense')
            data.loc[:,'ranks']=np.floor(data['rank']*bins/(n_unique+1))
            data.loc[np.isnan(data[var]),'ranks']=bins

        if is_object_dtype(data[var]):
            data.loc[:,'rank']=data[var]
            data.loc[pd.isnull(data[var]),'ranks']='Missing'


        binned_counts=data.groupby(['ranks',target_variable])['wgt'].sum().reset_index(name='count_y')
        