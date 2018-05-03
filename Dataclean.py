# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 13:49:55 2018

@author: Tim
"""

import pandas as pd
import numpy as np
import sys

dropna_df = pd.DataFrame(data=None)


input_filename = 'C:\Users\Asus\Desktop\Input Data\USD_Currency_Pairs $ (Expanded sample).csv'
data_df = pd.read_csv(input_filename)
columns = data_df.columns[0]
count=0
# Approach 1: Forward filling
"""for a in range(0, data_df.shape[1]):
	print data_df.columns[a]
	for i in range(0, data_df.shape[0]):
		if pd.isnull(data_df.iloc[i, a]):
			if i!=0:
				if not pd.isnull(data_df.iloc[i-1,a]):
					data_df.set_value(i,a, data_df.iloc[i-1,a], takeable=True)
					count+=1"""



# Approach 2: Deleting empty rows (how='all'), followed by forward filling
#data_df=data_df.iloc[:,1:]
print data_df.shape
data_df=data_df.dropna(axis=0, how='all', thresh=2)
print data_df.shape
data_df = data_df.reset_index(drop=True)
for a in range(0, data_df.shape[1]):
	print data_df.columns[a]
	for i in range(0, data_df.shape[0]):
		if pd.isnull(data_df.iloc[i, a]):
			if i!=0:
				if not pd.isnull(data_df.iloc[i-1,a]):
					data_df.set_value(i,a, data_df.iloc[i-1,a], takeable=True)
					count+=1

# Approach 2: Deleting empty rows (how='any')
"""data_df=data_df.iloc[:,1:]
data_df_drop=data_df
for i in range(0, data_df.shape[0]):
	#print pd.isnull(data_df.iloc[i,:]).sum()
	#print pd.isnull(data_df.iloc[i-1,:]).sum()
	if pd.isnull(data_df.iloc[i,:]).sum() > pd.isnull(data_df.iloc[i-1,:]).sum():
		print "Missing values detected at i=%s" %(i)
		#print data_df.iloc[i,:]
		#print data_df.iloc[i-1,:]
		data_df.iloc[i:,]=np.nan
		count+=1"""
		
data_df=data_df.dropna(axis=0, how='all')
data_df = data_df.reset_index(drop=True)


print count
writer = pd.ExcelWriter('USD_Currency_Pairs $ fill gaps.xlsx', engine='xlsxwriter')
data_df.to_excel(writer, sheet_name='Daily')
writer.save()