import numpy as np
import pandas as pd

def regression_summary(est,names, X,y):
	from statsmodels.stats.outliers_influence import variance_inflation_factor
	from sklearn.feature_selection import f_regression
	
	vif_list = list()
	for i in range(X.shape[1]):
		vif_list.append(variance_inflation_factor(X, i))

	tmp1=pd.DataFrame(np.array(names))
	F, pval = f_regression(X,y)
	tmp2=pd.DataFrame(est.coef_).T
	tmp3=pd.DataFrame(pval)
	tmp4=pd.DataFrame(vif_list)
	tmp5=pd.concat([tmp1,tmp2,tmp3,tmp4],axis=1)
	tmp5.columns = ['Variable_Name','Coef','p-value', 'VIF']
	# tmp5.to_csv('Regression_Summary.csv', index=False)
	return tmp5

def output_feature_importance(est,names,k=10):
	# sort importances
	indices = np.argsort(est.feature_importances_)[::-1] #argsorts in descending order
	
	tmp1=pd.DataFrame(np.array(names)[indices])
	tmp2=pd.DataFrame((est.feature_importances_[indices]))
	tmp3=pd.concat([tmp1,tmp2],axis=1)
	tmp3.columns = ['Variable_Name','Importance']
	return tmp3.iloc[:k,:]

def rankplot(dataset, indep_var, dep_var, tranches=10):
	"""
	Returns a dataframe containing mean of dependent variable grouped into tranches sorted by indep. variable
	Also return volume for each bin
	Missing or NAN values will be a bin on its own
	"""
	dtsn = dataset.ix[:,[dep_var,indep_var]]
	X = pd.DataFrame()
	
	if dataset[indep_var].dtype == np.float64 or dataset[indep_var].dtype == np.int64:		
		dtsn.sort_values(indep_var, ascending=False, inplace=True) #Will sort NA values to the end
		dtsn.reset_index(drop=True, inplace=True)
		# dtsn['bins'] = pd.qcut(dtsn[indep_var], tranches, labels=False)
		##alternative group by myself##
		# bin_intervals = np.linspace(0,dataset[indep_var].count(),tranches)
		### Create bins, left join with dataset, and fill NAN as its own category ###
		X['bins'] = pd.cut(range(dataset[indep_var].count()), tranches, labels=False)
		dtsn=dtsn.merge(X,how='left', left_index =True, right_index = True)
		dtsn['bins'] = dtsn['bins'].fillna(-1)
		group_obj=dtsn.groupby(['bins'])
		
		X = pd.concat([group_obj[dep_var].mean(),
						group_obj[dep_var].count(),
						group_obj[indep_var].min(),
						group_obj[indep_var].max(),
						group_obj[dep_var].sum()/dtsn[dep_var].sum()], axis=1)
		X.columns = list(['Mean_Dep_Var', 'NObs', 'Min_Indep_Var', 'Max_Indep_Var','Capture_Rate'])
		
	else:
		if len(dtsn[indep_var].unique()) <= 150:
			dtsn[indep_var] = dtsn[indep_var].fillna('MISSING')
			group_obj = dataset.groupby([indep_var])

			X = pd.concat([group_obj[dep_var].mean(),group_obj[dep_var].count()], axis=1)

			X.columns = list(['Mean_Dep_Var', 'NObs'])
		
	return X

def OOB_averages(df_train_raw, df_test_raw, response, method='mean'):
	"""
	### This function maps categorial columns into numeric variables by taking the mean of the response rate###
	### df_train and df_test are dataframe of all categorical features
	### Determine levels. Create vector of mean for each level ###	
	"""
	print 'Hashing categorical features....'
	train_result=pd.DataFrame()
	test_result=pd.DataFrame()

	df_train = df_train_raw.copy()
	df_test = df_test_raw.copy()

	# column_names=list(df_train.columns.values)

	if not df_train.empty:
		for col in df_train:
			df_col=df_train[col]
			df_col_test=df_test[col]
			
			levels=df_col.unique()

			print col
			print levels
			
			for lvl in levels:					
				lvl_mean=response[df_col==lvl].astype(int).mean()

				df_col = df_col.map(lambda x: lvl_mean if x == lvl else x)
				df_col_test = df_col_test.map(lambda x: lvl_mean if x == lvl else x)

			# df_col_test=pd.to_numeric(df_col_test,errors='coerce')

			train_result=pd.concat([train_result, df_col.astype(float)],axis=1)
			test_result=pd.concat([test_result, df_col_test.astype(float)],axis=1)
		return train_result, test_result
	else:
		return None, None