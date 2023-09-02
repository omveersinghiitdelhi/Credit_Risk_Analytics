import pandas as pd
import numpy as np
from Ipython.display import Markdown, display, HTML, display_html

def flooring_capping(data,var,floor_val=None,cap_val=None):
    """ Create transformed variable by flooring, capping or clipping based on the floor and cap values"""
    if floor_val==None and cap_val==None:
        raise ValueError("Both Floor and Cap values can not be None")
    elif floor_val==None:
        data[f'{var}']=data[var].clip(floor_val,cap_val)
    elif cap_val==None:
        data[f'{var}']=data[var].clip(floor_val,cap_val)
    else:
        data[f'{var}']=data[var].clip(floor_val,cap_val)

def log_vars(data, var, add_const=1):
    """Create log(x+constant) transformations"""
    if (data[var]+add_const).min(axis=0)<=0:
        added_constant=add_const+np.abs((data[var]+add_const).min())+0.0000001
        data[f'{var}']=np.log(data[var]+added_constant)
        display(Markdown("Added constant to log function: "+ f"{added_constant}"))
    else:
        data[f'{var}']=np.log(data[var]+add_const)

def replace_default(data,var,replacement_dict):
    """ Replace a default value in a variable with a special values"""
    data[f'{var}']=data[var].replace(replacement_dict)

def indicator_vars(data,var,query_condition,var_name_suffix='I'):
    """ Create Indicator variable based on the string of query condition"""
    indx=list(data.query(query_condition).index)
    data[f'{var}']=0
    data.loc[indx,f'{var}']=1

def numeric_binning(data, var, bins_split_points, right=True):
    """ Transforme continuous values to discrete labels by binning will be performed"""
    min_point = data[var].min()
    max_point = data[var].max()+0.00001
    bins=[min_point]+bins_split_points+[max_point]
    labels=list(range(1,len(bins)))
    for i in range(0,len(bins)-1):
        print(f"Bin {i+1} is {bins[i]}-{bins[i+1]} and its label is {labels[i]}")
    data[f'{var}']=pd.cut(data[var], bins=bins, labels=labels,include_lowest=True,right=right)
    data[f'{var}']=data[f'{var}'].astype('int64')

def woe_map_df(variable,mapping):
    """Create DataFrame summarising conversion to apply to feature. """
    outset=pd.DataFrame({'Variable':variable, 'WOE mapping':[mapping]})
    return outset

def woe_binning(data,var, bins_split_points,weight=None, relationship='N', right=True):
    """ Transform continuous values to discrete labels by binning values"""
    min_point = data[var].min()
    max_point = data[var].max()+0.00001   #Added 0.001 to include highest value
    bins=[min_point]+bins_split_points+[max_point]
    labels=list(range(1,len(bins)))
    for i in range(0,len(bins)-1):
        print(f"Bin {i+1} is {bins[i]}-{bins[i+1]} and its label is {labels[i]}")
    data[f'{var}']=pd.cut(data[var], bins=bins, labels=labels,include_lowest=True,right=right)
    
    if weight:
        woe_df1=data.copy(deep=True)
        woe_df1['wgt_y']=woe_df1['flow_default']*woe_df1['wgt']
        woe_df1=pd.DataFrame({'bad':woe_df1.groupby(f'{var}', dropna=False)['wgt_y'].sum(),
                              'total':woe_df1.groupby(f'{var}', dropna=False)['wgt'].sum()})
    else:
        woe_df1=pd.DataFrame({'bad':data.groupby(f'{var}', dropna=False)[target].sum(),
                              'total':data.groupby(f'{var}', dropna=False)[target].count()})
    woe_df1['good']=woe_df1['total']-woe_df1['bad']
    woe_df1['volume']=woe_df1['total']/woe_df1['total'].sum()
    woe_df1['bad_%']=woe_df1['bad']/woe_df1['bad'].sum()
    woe_df1['good_%']=woe_df1['good']/woe_df1['good'].sum()

    if relationship=='N':
        woe_df1['bad_%']=np.where(woe_df1['bad_%']==0,0.000001,woe_df1['bad_%'])
        woe_df1['WOE%']=np.log(woe_df1['good_%']/woe_df1['bad_%'])
    elif relationship=='P':
        woe_df1['good_%']=np.where(woe_df1['good_%']==0,0.000001,woe_df1['good_%'])
        woe_df1['WOE']=np.log(woe_df1['bad_%']/woe_df1['good_%'])
    WOE_mapping=pd.Series(woe_df1['WOE'], index=woe_df1.index).to_dict()
    print(WOE_mapping)
    data[f'{var}'].replace(WOE_mapping,inplace=True)

    woe_dictionary=woe_map_df(var,WOE_mapping)

    return woe_dictionary

def piecewise_vars(data,var, split_points_list):
    """Create piecewise transformed variables based on the split points"""
    min_point=data[var].min()
    max_point=data[var].max()
    counter=1

    for point in split_points_list:
        low_point_name='min_val' if counter==1 else min_point
        data[f'{var}']=np.where(data[var].between(min_point,point),data[var],
                                (np.where(data[var]<min_point,min_point,point)))
        min_point=point
        counter+=1
    data[f'{var}']=np.where(data[var].between(min_point,max_point),data[var],
                            (np.where(data[var]<min_point,min_point,max_point)))

def math_trans(data, var, trans_type='power', n=2):
    """ Function to apply different math transformations of the numeric variables"""
    if trans_type=='power':
        data[f'{var}']=np.power(data[var], n)
    if trans_type=='root_power':
        data[f'{var}']=np.where(data[var]<0,0,np.power(data[var],1/n))
    if trans_type=='inverse':
        data[f'{var}']=np.where(data[var]<0,0,1/data[var])
    if trans_type=='exp':
        data[f'{var}']=np.exp(data[var])

def cat_label_encoding(data,var, categories_labels_dict):
    """Perform label encoding for categorical variables"""
    for categories, label in categories_labels_dict.items():
        data.loc[data[var].isin(list(categories)), f'{var}']=label