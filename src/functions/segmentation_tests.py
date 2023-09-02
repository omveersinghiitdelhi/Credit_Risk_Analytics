"""
Script contains functions to test homogeneity and heterogeneity Chi Square and SoumersD of the subsamples
Author: Omveer Singh
Date : 10-10-2020
"""
from ast import Return
from cProfile import label
from cgi import test
from logging import _Level
from statistics import variance
from tkinter import N, SE
import pandas as pd
import numpy as np
import itertools
import scipy as sp
from scipy import stats
from scipy.stats import t
from scipy.stats import chi2_contingency
from scipy.stats._stats import _kendall_dis
from statsmodels.stats.weightstats as wt_stats
from matplotlib import pyplot

def chisqtest(df,test_var,test_val,group,target,unique_id=None,weight=None,random_seed=1234):
    """
    """
    #Filter by Values
    chisq_df=df.loc[df[test_var]==test_val].copy()
    #randomly split variable into groups:
    np.random.seed(random_seed)
    if unique_id:
        rand_df=pd.DataFrame({unique_id:chisq_df[unique_id].unique()})
        rand_df['random']=np.random.uniform(0,1,len(rand_df))
        rand_df['Rank']=rand_df['random'].rank()
        rand_df['group']=np.floor(rand_df['Rank']*group/(len(rand_df['Rank'])+1)).astype(int)+1
        chisq_df=pd.merge(chisq_df,rand_df, on=unique_id, how='left')
    else:
        chisq_df['random']=np.random.uniform(0,1,chisq_df.shape[0])
        chisq_df['Rank']=chisq_df['random'].rank()
        chisq_df['group']=np.floor(chisq_df['Rank']*group/(len(chisq_df['Rank'])+1)).astype(int)+1
    #Chi Square Contengency Table
    if weight:
        contingency = pd.crosstab(chisq_df['group'],chisq_df[target],chisq_df[weight],aggfunc=sum)
    else:
        contingency = pd.crosstab(chisq_df['group'],chisq_df[target])
    c,p,dof,expected=chi2_contingency(contingency)
    chisqoutcome=pd.DataFrame({'DF':dof,'Chisq Value':c,'Prob':p},index=[test_val])
    contingency['event_rate']=contingency[1]/(contingency[0]+contingency[1])
    return chisqoutcome, contingency
def chisqplot(contingency_tb, chart_title):
    """
    Plot Orthogonal Distance Regression
    """
    bar_width=0.4
    chart1=pyplot.bar(contingency_tb.index, contingency_tb[0], bar_width, label='0')
    chart2=pyplot.bar(contingency_tb.index+bar_width, contingency_tb[1], bar_width, label='1')
    pyplot.tight_layout()
    pyplot.title(chart_title)
    pyplot.xlabel('Bins')
    pyplot.ylabel("Counts")
    pyplot.legend()
    pyplot.show()

    pyplot.plot(contingency_tb.index, contingency_tb['event_rate'],marker='o')
    pyplot.xticks(contingency_tb.index)
    pyplot.xlabel("Bins")
    pyplot.ylabel("event_rate")
    pyplot.gca().set_ylim([0,1])
    pyplot.show

def clm_ttest(df,test_var,alpha,target,weight=None):
    """
    Heterogeneity Test - CI T Test
    """
    if weight:
        def weighted_avg(x,cols,w):
            return pd.Series(np.average(x[cols],weights=x[w], axis=0),[mean])
        def weighted_std(x,cols,w):
            average=np.average(x[cols],weights=x[w], axis=0)
            variance=np.average((x[cols]-average)**2,weights=x[w],axis=0)
            std=np.sqrt(variance)
            return pd.Series(std,['std'])
        cnt=df.groupby(test_var).agg(['count'])[weight]
        mean=df.groupby(test_var).apply(weighted_avg,[target],w=weight)
        std=df.groupby(test_var).apply(weighted_std,[target],w=weight)
        dfs=[cnt,mean,std]
        clm=dfs[0].join(dfs[1:])
    else:
        clm=df.groupby(test_var)[target].describe()['count','mean','std']
    clm['Lower CL']=clm['mean']+t.ppf(alpha/2,clm['count']-1)*clm['std']/np.sqrt(clm['count'])
    clm['Upper CL']=clm['mean']+t.ppf(1-alpha/2,clm['count']-1)*clm['std']/np.sqrt(clm['count'])
    return clm
def t_test(df,test_var,alpha,target,weight=None, equal_var=False):
    """
    Perform Heterogeneity test  by means of TTest
    """
    ttest=df.groupby(test_var)[target].describe()
    n=ttest.shape[0]
    values=df[test_var].unique().tolist()
    values.sort()
    var_nb=list(range(1,n+1))
    var_dict=dict(zip(var_nb,values))
    if n==1:
        print("Only one lavel - pairwise comparison cannot be performed")
        ttest_outcome=[]
    else:
        compare=list(itertools.combinations(list(range(1,n+1)),2))
        ttest_t,ttest_p,mean1,mean2,compare_val=[],[],[],[],[]
        for i in range(len(compare)):
            val1_df=df[df[test_var]==var_dict[compare[i][0]]]
            val2_df=df[df[test_var]==var_dict[compare[i][1]]]
            if weight:
                weights=(val1_df[weight],val2_df[weight])
            else:
                weights=(None,None)
            outcome=wt_stats.ttest_ind(val1_df[target],val2_df[target],weights=weights,usevar=('pooled' if equal_var else 'unequal'))
            ttest_t.append(outcome[0])
            ttest_p.append(outcome[1])
            mean1.append(ttest['mean'][var_dict[compare[i][0]]])
            mean2.append(ttest['mean'][var_dict[compare[i][1]]])
            compare_val.append((var_dict[compare[i][0]],var_dict[compare[i][1]]))
        ttest_outcome=pd.DataFrame({test_var:compare_val,'mean1':mean1,'mean2':mean2,'T STats':ttest_t,'p_value':ttest_p})
        ttest_outcome['conclusion']=np.where(ttest_outcome['p_value']<alpha,'Different Mean','Same Mean')
    return ttest_outcome

def chiqstest_heterogeneity(df,test_var,target,weight=None):
    """ Performs Heterogeneity testing by means of Chi-Square test    """
    if weight:
        contingency=pd.crosstab(df[test_var], df[target],df[weight],aggfunc=sum)
    else:
        contingency=pd.crosstab(df[test_var], df[target])
    c, p, dof, expected =chi2_contingency(contingency)
    chisqoutcome=pd.DataFrame({'DF':[dof],'Chisq Value':[c], 'Prob':[p]})
    contingency['event_rate']=contingency[1]/(contingency[0]+contingency[1])
    return chisqoutcome, contingency

def Gini_IV_for_Segment(data_name,seg_name,target):
    """ Segmentation Discrimatory Power - GINI & IV. Note that Gini calculated in this function is using the approximated AUC ROC approach"""
    Seg_tb=data_name.groupby([seg_name,target])[target].count().unstack(1)
    Seg_tb['ODR']=Seg_tb[1]/(Seg_tb[0]+Seg_tb[1])
    AUC_tb=Seg_tb.sort_values(by=['ODR'], ascending=True)
    AUC_tb['%Good']=AUC_tb[0]/AUC_tb[0].sum()
    AUC_tb['%Bad']=AUC_tb[1]/AUC_tb[1].sum()
    AUC_tb['%Good_Cum']=np.cumsum(AUC_tb['%Good'])
    AUC_tb['%Bad_Cum']=np.cumsum(AUC_tb['%Bad'])
    AUC_tb['%Bad_Cum_Lag']=AUC_tb['%Bad_Cum'].shift(periods=1)
    AUC_tb['%Bad_Cum_Lag']=np.where(np.isnan(AUC_tb['%Bad_Cum_Lag']),0,AUC_tb['%Bad_Cum_Lag'])
    AUC_tb['AUC']=AUC_tb['%Good']*(AUC_tb['%Bad_Cum']+AUC_tb['%Bad_Cum_Lag'])*0.5
    AUC_tb['IV_i']=(AUC_tb['%Good']-AUC_tb['%Bad'])*np.log(AUC_tb['%Good']/AUC_tb['%Bad'])
    GINI=AUC_tb['AUC'].sum()*2-1
    IV=AUC_tb['IV_i'].sum()
    return AUC_tb,GINI,IV

# SomersD using concordant discordant approach would be more accurate measure of Gini as the ROC AUC is approaximate area measure. Both should give very close GiNi

def SomersD(df, target_var, seg_name):
    """
    Segmentation Descriminatory Power - SomersD.
    """
    df=df[[seg_name,target_var]].copy()
    # Assign WOE for categorical variables
    woe=df.groupby([seg_name,target_var])[seg_name].count().unstack()
    woe['%Good']=woe[0]/woe[0].sum()
    woe['%Bad']=woe[1]/woe[1].sum()
    woe['woe']=np.log(woe['%Good']/woe['%Bad'])
    seg_woe_dict=woe['woe'].to_dict()
    df[seg_name].replace(seg_woe_dict)

    # Concordant Discordant Calculation
    x=df[target_var]
    y=df[seg_name]
    x=np.asarray(x).ravel()
    y=np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("All Input must be of same size,"
                         "found x-size %s and y-size %s" %(x.size,y.size))
    
    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt>1]
        return ((cnt*(cnt-1)//2).sum(),(cnt*(cnt-1)*(cnt-2)).sum(),(cnt*(cnt-1)*(2*cnt+5)))
    
    size=x.size
    perm=np.argsort(y) # sort on y and convert y to dense ranks

    x,y = x[perm],y[perm]
    y=np.r_[True,y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # Stable sort on x and convert x to dense ranks
    perm=np.argsort(x,kind='mergesort')
    x,y = x[perm],y[perm]
    x=np.r_[True,(x[1:]!=x[:-1]) | (y[1:]!=y[:-1]),True]
    
    dis=_kendall_dis(x,y)   #discordant pairs
    obs = np.r_[True,(x[1:] != x[:-1]) | (y[1:] != y[:-1]),True]
    cnt=np.diff(np.where(obs)[0]).astype('int64', copy=False)

    ntie = (cnt*(cnt-1)//2).sum()   #joint ties
    xtie, x0, x1 =count_rank_tie(x) # ties in x, stats
    ytie, y0, y1 =count_rank_tie(y) #ties in y, stats

    tot = (size *(size -1))//2
    SD = (tot - xtie -ytie + ntie -2*dis)/(tot -xtie)
    tau_a=(tot - xtie -ytie + ntie -2*dis)/(tot)  #Kendalls tau-a
    tau_b=(tot - xtie -ytie + ntie -2*dis)/np.sqrt(tot -xtie)/np.sqrt(tot -ytie) #Kendalls tau-b

    return (SD,tau_a,tau_b)

def chisqtest_pairwise(df,test_var,test_val,group,target,unique_id=None,weight=None,random_seed=1234):
    """ Performs the pairwise homogeneity testing by means of Chi-Square test"""
    #Filter  by values
    chisq_df=df.loc[df[test_var]==test_val].copy()

    #randomly split variable into groups
    np.random.seed(random_seed)
    if unique_id:
        rand_df=pd.pd.DataFrame({unique_id:chisq_df[unique_id].unique()})
        rand_df['random']=np.random.uniform(0,1,len(rand_df))
        rand_df['Rank']=rand_df['random'].rank()
        rand_df['group'] = np.floor(rand_df['Rank']*group/(len(rand_df['Rank'])+1))
        chisq_df=pd.merge(chisq_df,rand_df, on=unique_id, how='left')
    else:
        chisq_df['random'] = np.random.uniform(0,1,chisq_df.shape[0])
        chisq_df['Rank'] = chisq_df['random'].rank()
        chisq_df['group']=np.floor(chisq_df['Rank']*group/(len(chisq_df['Rank'])+1))
    
    # loop all pairs
    pair =list(itertools.combinations(list(range(group)),2))
    chisq_all = pd.DataFrame()

    for i in pair:
        pair_data=chisq_df[chisq_df['group'].isin(i)]
        # Chi- Square contingency Table
        if weight:
            contingency = pd.crosstab(pair_data['group'],pair_data[target],pair_data[weight],aggfunc=sum)
        else:
            contingency = pd.crosstab(pair_data['group'],pair_data[target])
        c, p, dof, expected =chi2_contingency(contingency)
        chisqoutcome=pd.DataFrame({'DF':dof,'Chisq Value':c,'Prob':p}, index=[i])
        contingency['event_rate'] = contingency[1]/(contingency[0]+contingency[1])
        chisq_all = chisq_all.append(chisq_all)
    return chisq_all