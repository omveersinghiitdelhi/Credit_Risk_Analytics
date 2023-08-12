import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr

def stressed_dfs(df_raw,df_transformed,target,model_variables,stressed_variable, stresses, weight=None):
	"""
	Create a dictionary containing DataFrames with Raw stressed variable and all other variables
	with transformation applied. Required raw and transformed Data Frame
	"""
	stressed_dfs = {}
	cols = [target]+([weight] if weight else [])+model_variables
	stressed_dfs[0]=df_transformed[col].copy()
	cols.remove(stressed_variable)
	for stress in stresses:
		stressed_df[stress]=df_transformed[cols].copy()
		stressed_df[stress][stressed_variable]=df_raw[stressed_variable]
	return stressed_dfs

def stressed_scores(stresses):
	"""
	Create an empty dictionary for stressed scored
	"""
	stressed_scores={}
	stressed_scores['variable']=[]
	stressed_scores[0]=[]
	for stress in stresses:
		stressed_scores[stress]=[]
	return stressed_scores

def stressed_variables(stresses):
	"""
	create empty dictionary for stressed transformed variable values.
	"""
	stressed_vars={}
	stressed_vars['Variable']=[]
	stressed_vars[0]=[]
	for stress in stresses:
		stressed_vars[stress]=[]
	return stressed_vars


def score_model(df,model,model_variables,prediction='predicted'):
	df=sm.add_constant(df,has_constant='add')
	df[prediction]=model.predict(df[['const']+ model_variables])
	df.drop(['const'], axis=1,inplace=True)
	return df

def rank_shift(df, variable, target, numeric_bin=None, drop_nans=True, weight=None, min_volume=0.0001):
	"""
	Create dictionary containing a mapping to the value of the next risk up/down for each bin/level.
	"""
	if df[variable].dtype == object or not numeric_bin:
		group_var=variable
		shift=variable
	else:
		group_var=numeric_bin
		shift='mean'
	if weight:
		wm=lambda x: np.average(x,weights=df.loc[x.index,weight])
		_grouped=df.groupby(group_var,dropna=drop_nans).agg(bad_rate=(target,wm), volume=(weight,'sum')).sort_values(['bad_rate']).reset_index()
		if group_var == numeric_bin:
			_var_mean=df.groupby(group_var, dropna=drop_nans).agg(mean=(variable,wm)).reset_index()
			_grouped=_grouped.merge(_var_mean, on=[group_var], how='left')
	else:
		_grouped=df.groupby(group_var,dropna=drop_nans).agg(bad_rate=(target,'mean'), volume=(weight,'sum')).sort_values(['bad_rate']).reset_index()
		if group_var == numeric_bin:
			_var_mean=df.groupby(group_var, dropna=drop_nans).agg(mean=(variable,'mean')).reset_index()
			_grouped=_grouped.merge(_var_mean, on=[group_var], how='left')
	_grouped['volume_perc']= _grouped['volume']/(_grouped['volume'].sum())
	# get variable correlation with target to determine which direction stress should be in
	_grouped.sort_values(['bad_rates'], ascending =False, inplace=True)
	# Do not shift very low volume groups or bad_rate == 0
	_grouped=_grouped[_grouped['volume_perc'] > min_volume]
	_grouped=_grouped[_grouped['bad_rate'] > 0]
	
	# shift up/down of the risk rank imputing value to the mean of the next risk rank
	_grouped['up']=_grouped[shift].shift(-1).fillna(_grouped[shift])
	_grouped['down']=_grouped[shift].shift(1).fillna(_grouped[shift])

	up_val=_grouped.set_index(group_var).to_dict()['up']
	down_val=_grouped.set_index(group_var).to_dict()['down']
	
	return up_val,down_val


def group_sample(df,variable,perc_shift,random_seed, numeric_bin=None, weight=None):
	"""
	Get random sample of the variable by groups
	"""
	_df=df[[variable]+([numeric_bin] if numeric_bin else [])+([weight] if weight else [])].copy()
	_df_samp = _df.groupby((numeric_bin if numeric_bin else variable), as_index=False).apply(lambda x: x.sample(frac=perc_shift,random_state=random_seed,weights=(weight if weight else None))).droplevel(0).sort_index()
	return _df_samp
	
	
def sensitivity_analysis(
	df_raw,df_transformed,model_variables,reg_model,final_transformations,target,continuous_stress,continuous_rank_stress,categorical_stress,binary_stress,random_seed,df_benchmark=None,test_stress_values=False,sensitivity_stresses=[0.05,0.10,0.15,0.20],smv_dict=None,weight=None,continuous_zeros=None,incl	ude_missings=None,no_bins=10,
):
	""" 
	Function to performs a sensitivity analysis for a linear model.
	"""
	
	all_stresses={}
	for d in (continuous_stress,continuous_rank_stress,categorical_stress,binary_stress):
		all_stresses.update(d)
	if not include_missings:
		include_missings=[]
	stresses=(test_stresses if test_stress_values else sensitivity_stresses	)
	_stresses=[-i for i in stresses[::-1]]+stresses
	_stressed_scores = stressed_scores(_stresses)
	_stressed_vars = stressed_variables(_stresses)
	_random_seeds = [random_seed+(n*n*500) for n, i in enumerate(_stresses)]

	# Baseline Score
	baseline_df = score_model(df_transformed, reg_model, model_variables)
	baseline_score =np.average(baseline_df['predicted'],weights=baseline_df['weight']) if weight else baseline_df['predicted'].mean()
	
	for var, stress_val in all_stresses.items():
		# Apply stress on raw variable except categorical variables.
		use_transformed=True if var in categorical_stress.keys() else False
		if use_transformed:
			_stressed_dfs=stressed_dfs(df_transformed,df_transformed,target,model_variables,var,_stresses,weight)
		else:
			_stressed_dfs=stressed_dfs(df_raw,df_transformed,target,model_variables,var,_stresses,weight)
		_stressed_scores['Variable'].append(var)
		_stressed_scores[0].append(baseline_score)
		_stressed_vars['Variable'].append(var)
		_stressed_vars[0].append(df_transformed[var].mean())
		_stress_vals=_stresses if test_stress_values else [i*stress_val for i in _stresses]
		
		### Continuous Features
		if var in continuous_stress.keys():
			# Get variable correlation
			_relationship=np.sign(spearmanr(df_raw[df_raw[var].notna()][var],df_raw[df_raw[var].notna()][target])[0])
			if _relationship == -1:
				_stress_vals = list(reversed(_stress_vals))
			# Continuous_zeros
			if var in continuous_zeros:
				_df_zeros = (df_raw[~df_raw[var].isin(smv_dict[var])] if var in list(smv_dict.keys()) else df_raw)
			_df_gt0 = _df_zeros[_df_zeros[var] > 0][[var,target]+([weight] if weight else [])]
			_df_le0 = _df_zeros[_df_zeros[var] <= 0][[var,target]+([weight] if weight else [])]
			# impute to median 
			gt0_impute=_df_gt0[var].median()
			le0_impute=_df_le0[var].median()
			# Calculate proportion of observations to  shift to shift in /out <=0 group
			gt0_size=(_df_gt0[weight].sum() if weight else len(_df_gt0))
			le0_size=(_df_le0[weight].sum() if weight else len(_df_le0))
			_zeroes_stress = [(abs(i) if n <= (len(_stresses)/2)-1 else abs(((le0_size*i)/gt0_size))) for n,i in enumerate(_stress_vals)]
		for n, stress in enumerate(_stresses):
			_df = _stressed_dfs[stress]		

			# Continuous Zeros
			if var in continuous_zeros:
				# if negative stress then shift x% random observations from <=0 for positive relationship for var and target else positive stress will shift varibales to >0
				if n <= (len(_stresses)/2)-1:
					_df_samp = _df_le0.sample(frac=_zeroes_stress[n],random_state=random_seeds[n],weights=(weight else None))
					_df[var] = np.where(_df.index.isin(_df_samp.index),gt0_impute,_df[var])
				else:
					_df_samp = _df_gt0.sample(frac=_zeroes_stress[n],random_state=random_seeds[n],weights=(weight else None))
					_df[var] = np.where(_df.index.isin(_df_samp.index),le0_impute,_df[var])
			# ignore any special missing vaalues
			if var in list(smv_dict.keys()):
				_df[var] = np.where(_df[var].isin(smv_dict[var]),_df[var],_df[var]*(1+_stress_vals[n]))
			else:
				_df[var] = _df[var]*(1+_stress_vals[n])

			#  Transform stressed variables
			_df = final_transformations(_df,var)

			# Score Model
			_df = score_model(_df,reg_model, model_variables)

			_stressed_scores[stress].append(np.average(_df['predicted'],weights=_df[weight]) if weight else _df['predicted'].mean())
			_stressed_scores[stress].append(np.average(_df[var],weights=_df[weight]) if weight else _df[var].mean())

		# Shift continuous values to the next risk group
		if var in continuous_rank_stress.keys():
			# Create Bins
			for stress in _stresses:
				_df = _stressed_dfs[stress]
				_df['bin'] = pd.cut((_df[~_df[var].isin(smv_dict[var])] if var in list(smv_dict.keys()) else _df)[var].rank(method='min'),bins=no_bins, labels=False).rank(method='dense')
			# Shift up/down risk rank - imputing value to the mean of next risk rank
			up_val, down_val =rank_shift(_stressed_dfs[_stresses[0]],var, target, numeric_bin='bin', drop_nans=(False if var in include_missings else True), weight=weight)
			for n , stress in enumerate(_stresses):
				_df = _stressed_dfs[stress]
				# Select x% random accounts to shift category
				_df_samp = group_sample((_df[~_df[var].isin(smv_dict[var])] if var in list(smv_dict.keys()) else _df),var,abs(_stress_vals[n]), _random_seeds[n], numeric_bin = 'bin', weight=weight)

				#if negative stress then shift values up a risk categories
				if  n < = (len(_stresses)/2)-1:
					_df[var] = np.where(_df.index.isin(_df.samp.index), _df['bin'].map(up_val), _df[var])
				#if positive stress then shift values down a risk categories
				else:
					_df[var] = np.where(_df.index.isin(_df.samp.index), _df['bin'].map(down_val), _df[var])
				# Transform the stressed variables
				_df = final_transformations(_df,var)
				# Score Model
				_df = score_model(_df,reg_model, model_variables)
				_stressed_scores[stress].append(np.average(_df['predicted'],weights=_df[weight]) if weight else _df['predicted'].mean())
				_stressed_scores[stress].append(np.average(_df[var],weights=_df[weight]) if weight else _df[var].mean())
		# For var in Categorical/Binary Features
		if var in list(categorical_stress.keys())+list(binary_stress.keys()):
			# Shift up down risk categories based on the default rate
			_df = (df_transformed if use_transformed else df_raw)
			up_val, down_val =rank_shift(_stressed_dfs[_stresses[0]],var, target, drop_nans=(False if var in include_missings else True), weight=weight, min_volume=0.0001)
			for n , stress in enumerate(_stresses):
				_df = _stressed_dfs[stress]
				# Select x% random accounts to shift category
				_df_samp = group_sample((_df[~_df[var].isin(smv_dict[var])] if var in list(smv_dict.keys()) else _df),var,abs(_stress_vals[n]), _random_seeds[n], weight=weight)
				#if negative stress then shift values up a risk categories
				if  n < = (len(_stresses)/2)-1:
					_df[var] = np.where(_df.index.isin(_df.samp.index), _df[var].map(up_val), _df[var])
				#if positive stress then shift values down a risk categories
				else:
					_df[var] = np.where(_df.index.isin(_df.samp.index), _df[var].map(down_val), _df[var])
				# Transform the stressed variables
				_df = final_transformations(_df,var)
				# Score Model
				_df = score_model(_df,reg_model, model_variables)
				_stressed_scores[stress].append(np.average(_df['predicted'],weights=_df[weight]) if weight else _df['predicted'].mean())
				_stressed_scores[stress].append(np.average(_df[var],weights=_df[weight]) if weight else _df[var].mean())
	
	# Aggregate Results
	_out_scores = pd.DataFrames(_stressed_scores)
	_out_scores = _out_scores[[_out_vars.columns[0]]+sorted(_out_scores.columns[1:].tolist())]
	_out_vars = pd.DataFrames(_stressed_scores)
	_out_vars = _out_vars[[_out_vars.columns[0]]+sorted(_out_vars.columns[1:].tolist())]
	
	# If Benchmark Data Frame is provided,  then append score and variable averages to stress data

	if df_benchmark is not None:
		_out_scores['benchmark_stress'] = (np.average(df_benchmark['predicted'],weights=df_benchmark[weight]) if weight else df_benchmark['predicted'].mean())
		var_benchmarks =[]
		for var in _out_vars['Variables'].tolist():
			var_benchmarks.append((np.average(df_benchmark[var],weights=df_benchmark[weight]) if weight else df_benchmark[var].mean()))
		_out_vars['benchmark_stress'] = var_benchmarks
	return _out_scores, _out_vars
		