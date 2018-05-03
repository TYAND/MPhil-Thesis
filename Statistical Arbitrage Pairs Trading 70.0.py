version="67.0"

import os
import time
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style('ticks')
sns.despine()
plt.rcParams["figure.figsize"] = (16,16)

start=time.clock()
HHMM = time.strftime("%H%M")
HHMMSS = time.strftime("%H:%M:%S")

print("""\

  ____    _    ____ _____ 
 / ___|  / \  |  _ \_   _|
 \___ \ / _ \ | |_) || |  
  ___) / ___ \|  __/ | |  
 |____/_/   \_\_|    |_|
 
 Statistical Arbitrage Pairs Trading                         
 
 
 Version:  %s
 Author:   Tim Yandle
________________________________________________________
                    """) % version
print
print "Time: %s" %(time.strftime("%H:%M:%S"))
print
print
print "Importing modules."

# Used for ADF testing, EG cointegration testing and running regressions
import statsmodels.tsa.stattools as ts

# Used to generate tuples for all possible unique combinations of the input
import itertools

# Interface between R and Python (enables usage of R libraries)
import rpy2.robjects as ro
r=ro.r

# Correlation testing
from scipy.stats.stats import pearsonr

# Multiple testing
from arch.bootstrap import StepM

# Multiple testing
from arch.bootstrap import SPA

# Progress bar
from tqdm import tqdm

# Text to speech
from tts_watson.TtsWatson import TtsWatson
ttsWatson = TtsWatson('21be4c9c-5a98-4acd-996d-e9bea794536e','FSQsw3dkvYul', 'en-US_AllisonVoice')

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print "All modules imported successfully."

def quickload():
# Loads data from hardcoded source
# User options are marked by ~'s and need to be inserted in the code directly
	
	overview_df = pd.DataFrame(data=None, dtype=object)
	overview_df=overview_df.T
	overview_df.set_value('Version', ' ', version)	
	overview_df.set_value('Date', ' ', time.strftime("%b %d"))
	overview_df.set_value('Time', ' ', time.strftime("%H:%M:%S"))


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	# Input data must be in .csv format
	# Y-axis should be time (oldest first), x-axis should be assets
	# Check the .csv file in notepad to make sure there aren't blank spaces at the bottom
	input_filename = 'C:\Users\Asus\Desktop\Input Data\USD_Currency_Pairs $ (23, 10k centered, droprows & fwd fill).csv'
	data_df = pd.read_csv(input_filename)
	overview_df.set_value('Input data', ' ', input_filename)
	
	# All of the columns between the two which the user enters will be selected
	asset_from='BZUS'
	asset_to='VZUS'
	assets_df = data_df.loc[:,asset_from:asset_to]
	if assets_df.shape[0] == data_df.shape[0]:	
		overview_df.set_value('Assets', ' ', 'All')
	else:
		from_to = "From " + asset_from + " to " + asset_to
		overview_df.set_value('Assets', ' ', from_to)
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	print "Data '%s' loaded successfully." % (input_filename)
	
	# Calculating the number of assets (and therefore the number of pairs) from the shape of the data frame
	shape = assets_df.shape
	num_obs = shape[0]*shape[1]
	obs_per_asset = shape[0]
	num_assets = shape[1]
	num_pairs = ((num_assets*(num_assets-1))/2)
	
	overview_df.set_value('Number of assets', ' ', num_assets)	
	overview_df.set_value('Number of observations', ' ', num_obs)	
	overview_df.set_value('Number of observations (per asset)', ' ', obs_per_asset)	
	overview_df.set_value('Number of pairs', ' ', num_pairs)	
	
	print
	print overview_df[5:]
	
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	# If set to Y, forces the ADF, correlation and cointegration tests to use the same p-values
	same_p_values = 'y'
	same_p_values = same_p_values.upper()
	
	if same_p_values == "N":
		input_pval_adf_lst = [0.10]
		overview_df.set_value('ADF p-value thresholds', ' ', input_pval_adf_lst)
		input_pval_corr_lst = [0.10]
		overview_df.set_value('Correlation p-value thresholds', ' ', input_pval_corr_lst)
		# EG works with any user-specified p-vaulues. PO only works with p-values of 0.01, 0.05 and 0.10
		input_pval_coint_lst = [0.10]
		overview_df.set_value('Cointegration p-value thresholds', ' ', input_pval_coint_lst)
	
	if same_p_values == "Y":
		input_pval_lst = [0.01, 0.05, 0.10]
		overview_df.set_value('ADF p-value thresholds', ' ', input_pval_lst)
		overview_df.set_value('Correlation p-value thresholds', ' ', input_pval_lst)
		overview_df.set_value('Cointegration p-value thresholds', ' ', input_pval_lst)
	
	# Code can't deal with window sizes that aren't a factor of the assets.df data frame length
	# Possible window sizes for USD_Currency_Pairs (Big 7) $ fill gaps 2.0 are 6125, 1225, 875, 245, 175 and 125
	# Possible window sizes for USD_Currency_Pairs (7, divisible) are 100, 200, 250, 300, 400, 500, 600, 750, 800, 1000
	# Possible window sizes for USD_Currency_Pairs (23, divisible, droprows) are 100, 115, 125, 230, 250, 460, 500, 575, 1150
	# Possible window sizes for USD_Currency_Pairs $ (23, ultra divisible, droprows & fwdfill) are 100, 125, 200, 250, 400, 500, 625, 1000, 1250, 2000, 2500

	window_size_lst = [250, 500]
	
	overview_df.set_value('Window sizes', ' ', window_size_lst)
	
	# If set to Y, enables the user to specify each set of triggers rather than automatically using all possible combinations of the inputs
	advanced_trigger_control = "y"
	advanced_trigger_control = advanced_trigger_control.upper()
	
	
	# P - stop-loss based on profit (losses)
	# PD - stop-loss based on profit (losses), and an additional open trigger condition which requires that disequlibrium not be excessively high
	# N - no stop loss
	stop_loss_type = ['pd', 'n']
	for i in range(0, len(stop_loss_type)):
		stop_loss_type[i]=stop_loss_type[i].upper()

	
	if advanced_trigger_control == "N": 
		open_trigger_lst=[2]
		overview_df.set_value('Open trigger sensitivity', ' ', open_trigger_lst)	
		close_trigger_lst=[1]
		overview_df.set_value('Close trigger sensitivity', ' ', close_trigger_lst)	
		stop_loss_trigger_lst = [3]
		if len(stop_loss_type) == 1 and stop_loss_type[0] == 'N':
			stop_loss_trigger_lst = ['N/A']
		overview_df.set_value('Stop loss trigger sensitivity', ' ', stop_loss_trigger_lst)
		overview_df.set_value('Stop loss type', ' ', stop_loss_type)
	
	if advanced_trigger_control == "Y":
		open_trigger_lst = []
		close_trigger_lst = []
		stop_loss_trigger_lst = []
		trigger_combinations = [[2,0,3], [2,1,3]]
		for i in range(0, len(trigger_combinations)):
			open_trigger_lst.append(trigger_combinations[i][0])
			close_trigger_lst.append(trigger_combinations[i][1])
			stop_loss_trigger_lst.append(trigger_combinations[i][2])
		overview_df.set_value('Open trigger sensitivity', ' ', open_trigger_lst)
		overview_df.set_value('Close trigger sensitivity', ' ', close_trigger_lst)
		if len(stop_loss_type) == 1 and stop_loss_type[0] == 'N':
			stop_loss_trigger_lst = ['N/A']
		overview_df.set_value('Stop loss trigger sensitivity', ' ', stop_loss_trigger_lst)
		overview_df.set_value('Stop loss type', ' ', stop_loss_type)
	# Options for the below setttings are all Y or N, unless otherwise stated
	
	# When asymmetric legs is on, the size of each leg in a given trade is scaled using the cointegrating relation (such that the sum of the legs is $2.00)
	# When asymmetric legs is off, the size of each leg in a given trade is $1.00 (sum of the legs is $2.00)
	asymmetric_legs = ['y']
	for i in range(0, len(asymmetric_legs)):
		asymmetric_legs[i] = asymmetric_legs[i].upper()
	overview_df.set_value('Asymmetric legs', ' ', asymmetric_legs)
	
	# If drop ADF is on, assets which fail ADF testing are excluded from further analysis for that window are excluded from further analysis window and, consequently, can't be traded in the next window
	# ADF test failure means that the asset is either stationary in levels or non-stationary in first differences (or both) - I(1)-ness is a nessecary precondition for cointegration between two series 
	drop_adf = ['y']
	for i in range(0, len(drop_adf)):
		drop_adf[i]=drop_adf[i].upper()
	overview_df.set_value('Drop ADF failures', ' ', drop_adf)
	
	# If correlation pre-screen is turned on, all elegible pairs of assets are tested for correlation
	# Pairs which aren't correlated are excluded from further analysis for that window and, consequently, can't be traded in the next window
	corr_prescreen = ['y']
	for i in range(0, len(corr_prescreen)):
		corr_prescreen[i]=corr_prescreen[i].upper()
	overview_df.set_value('Correlation pre-screen', ' ', corr_prescreen)
	
	# Options: PO or EG
	# PO: Computes the Phillips-Ouliaris test for the null hypothesis that x and y are not cointegrated
	# EG: Computes the augmented Engle-Granger two-step test for the null that x and y are not cointegrated
	# EG works with any user-specified p-vaulues. PO only works with p-values of 0.01, 0.05 and 0.10
	# When using PO remeber to manually set the sub-variant (PU or PZ)
	coint_test = ['po']
	for i in range(0, len(coint_test)):
		coint_test[i]=coint_test[i].upper()
	overview_df.set_value('Cointegration test', ' ', coint_test)
	
	# If include constant is turned on, a constant is included in the ADF and cointegration test specification 
	# Also includes a constant in the regression which estimates the cointegration relation and the calculation of disequlibrium
	incl_constant = ['n']
	for i in range(0, len(incl_constant)):
		incl_constant[i]=incl_constant[i].upper()
	overview_df.set_value('Include constant', ' ', incl_constant)
	
	# If yes, saves the results to an excel file 
	save = 'y'
	save=save.upper()
	overview_df.set_value('Save', ' ', save)
	
	# If yes, saves every time the analysis for a model is completed 
	safe_save = 'n'
	safe_save=safe_save.upper()
	overview_df.set_value('Safe save', ' ', safe_save)
	
	# If yes, saves the results of the ADF, correlation and cointegration tests as well as trade logs and disequlibrium measurements for every model
	save_details = 'y'
	save_details=save_details.upper()
	overview_df.set_value('Save details', ' ', save_details)
	
	# If yes, drops rows or columns for which there is no data before saving
	drop_empties = 'n'
	drop_empties=drop_empties.upper()
	overview_df.set_value('Drop empties', ' ', drop_empties)
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	# Checks if open triggers are both greater than close triggers and smaller than stop loss triggers
	if advanced_trigger_control == "N":
		for x in open_trigger_lst:
			if all(i < x for i in close_trigger_lst) == False:
				print
				print "Warning: Triggers set incorrectly"
				print
				sys.exit()
			if all(i > x for i in stop_loss_trigger_lst) == False:
				print
				print "Warning: Triggers set incorrectly"
				print
				sys.exit()
				
	# Checks that all window sizes are factors of the assets_df length
	for i in window_size_lst:
		if not assets_df.shape[0] % i == 0:
			print
			print "Warning: Window size set incorrectly (%s is not divisible by %s)" % (assets_df.shape[0], i)
			print
			sys.exit()
	
	# If cointegration is set to PO, checks that the p-values are valid
	if "PO" in coint_test:
		x = [0.01, 0.05, 0.10]
		if same_p_values == "Y":
			z = input_pval_lst
		if same_p_values == "N":
			z = input_pval_coint_lst
		if any(i not in x for i in z):
			print
			print "Warning: P-values set incorrectly (PO can only use 0.01, 0.05 and 0.10, EG can use any)"
			print
			sys.exit()
	
	z = ['P', 'PD', 'N']
	for x in stop_loss_type:
		if not x in z:
			print
			print " Warning: Stop loss type set incorrectly (%s is not a valid input)" % (x)
			print
			sys.exit()
		 		   
	print
	print overview_df[9:]
	
	models_df = pd.DataFrame(data=None)
	overview_df.set_value('Number of models', ' ', 0)
	
	if advanced_trigger_control == "Y":
		if same_p_values == "Y":
			num_models = len(input_pval_lst)*len(window_size_lst)*len(trigger_combinations)*len(asymmetric_legs)*len(drop_adf)*len(corr_prescreen)*len(coint_test)*len(incl_constant)*len(stop_loss_type)
			setting_combinations_lst = list(itertools.product(input_pval_lst, ['N/A'], ['N/A'], window_size_lst, trigger_combinations, ['N/A'], ['N/A'], asymmetric_legs, drop_adf, corr_prescreen, coint_test, incl_constant, stop_loss_type))
			for i in range(0,num_models):
				models_df.set_value('ADF p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[0])
				models_df.set_value('Correlation p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[0])
				models_df.set_value('Cointegration p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[0])
				models_df.set_value('Window size', 'Model ' + str(i), (setting_combinations_lst[i])[3])
				models_df.set_value('Open trigger sensitivity', 'Model ' + str(i), (setting_combinations_lst[i])[4][0])
				models_df.set_value('Close trigger sensitivity', 'Model ' + str(i), (setting_combinations_lst[i])[4][1])
				models_df.set_value('Stop loss trigger sensitivity', 'Model ' + str(i), (setting_combinations_lst[i])[4][2])
				lst = list(setting_combinations_lst[i])
				lst[1] = lst[0]
				lst[2] = lst[0]
				lst[5] = lst[4][1]
				lst[6] = lst[4][2]
			   	lst[4] = lst[4][0]
				setting_combinations_lst[i]=tuple(lst)
		if same_p_values == "N":
			num_models = len(input_pval_adf_lst)*len(input_pval_corr_lst)*len(input_pval_coint_lst)*len(window_size_lst)*len(trigger_combinations)*len(asymmetric_legs)*len(drop_adf)*len(corr_prescreen)*len(coint_test)*len(incl_constant)*len(stop_loss_type)
			setting_combinations_lst = list(itertools.product(input_pval_adf_lst, input_pval_corr_lst, input_pval_coint_lst, window_size_lst, trigger_combinations, ['N/A'], ['N/A'], asymmetric_legs, drop_adf, corr_prescreen, coint_test, incl_constant, stop_loss_type))
			for i in range(0,num_models):
				models_df.set_value('ADF p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[0])
				models_df.set_value('Correlation p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[1])
				models_df.set_value('Cointegration p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[2])
				models_df.set_value('Window size', 'Model ' + str(i), (setting_combinations_lst[i])[3])
				models_df.set_value('Open trigger sensitivity', 'Model ' + str(i), (setting_combinations_lst[i])[4][0])
				models_df.set_value('Close trigger sensitivity', 'Model ' + str(i), (setting_combinations_lst[i])[4][1])
				models_df.set_value('Stop loss trigger sensitivity', 'Model ' + str(i), (setting_combinations_lst[i])[4][2])
				lst = list(setting_combinations_lst[i])
				lst[5] = lst[4][1]
				lst[6] = lst[4][2]
			   	lst[4] = lst[4][0]
				setting_combinations_lst[i]=tuple(lst)

		
	if advanced_trigger_control == "N":
		if same_p_values == "Y":
			num_models = len(input_pval_lst)*len(window_size_lst)*len(open_trigger_lst)*len(close_trigger_lst)*len(stop_loss_trigger_lst)*len(asymmetric_legs)*len(drop_adf)*len(corr_prescreen)*len(coint_test)*len(incl_constant)*len(stop_loss_type)
			setting_combinations_lst = list(itertools.product(input_pval_lst, ['N/A'], ['N/A'], window_size_lst, open_trigger_lst, close_trigger_lst, stop_loss_trigger_lst, asymmetric_legs, drop_adf, corr_prescreen, coint_test, incl_constant, stop_loss_type))
			for i in range(0,num_models):
				models_df.set_value('ADF p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[0])
				models_df.set_value('Correlation p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[0])
				models_df.set_value('Cointegration p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[0])
				models_df.set_value('Window size', 'Model ' + str(i), (setting_combinations_lst[i])[3])
				lst = list(setting_combinations_lst[i])
				lst[1] = lst[0]
				lst[2] = lst[0]
				setting_combinations_lst[i]=tuple(lst)
		if same_p_values == "N":
			num_models = len(input_pval_adf_lst)*len(input_pval_corr_lst)*len(input_pval_coint_lst)*len(window_size_lst)*len(open_trigger_lst)*len(close_trigger_lst)*len(stop_loss_trigger_lst)*len(asymmetric_legs)*len(drop_adf)*len(corr_prescreen)*len(coint_test)*len(incl_constant)*len(stop_loss_type)
			setting_combinations_lst = list(itertools.product(input_pval_adf_lst, input_pval_corr_lst, input_pval_coint_lst, window_size_lst, open_trigger_lst, close_trigger_lst, stop_loss_trigger_lst, asymmetric_legs, drop_adf, corr_prescreen, coint_test, incl_constant, stop_loss_type))
			for i in range(0,num_models):
				models_df.set_value('ADF p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[0])
				models_df.set_value('Correlation p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[1])
				models_df.set_value('Cointegration p-value threshold', 'Model ' + str(i), (setting_combinations_lst[i])[2])
				models_df.set_value('Window size', 'Model ' + str(i), (setting_combinations_lst[i])[3])
		for i in range(0,num_models):
			models_df.set_value('Open trigger sensitivity', 'Model ' + str(i), (setting_combinations_lst[i])[4])
			models_df.set_value('Close trigger sensitivity', 'Model ' + str(i), (setting_combinations_lst[i])[5])
			models_df.set_value('Stop loss trigger sensitivity', 'Model ' + str(i), (setting_combinations_lst[i])[6])
				
			
	for i in range(0,num_models):
		models_df.loc['Stop loss type', 'Model ' + str(i)] = (setting_combinations_lst[i])[12]
		models_df.loc['Asymmetric legs', 'Model ' + str(i)] = (setting_combinations_lst[i])[7]
		models_df.set_value('Drop ADF failures', 'Model ' + str(i), (setting_combinations_lst[i])[8])
		models_df.set_value('Correlation pre-screen', 'Model ' + str(i), (setting_combinations_lst[i])[9])
		models_df.set_value('Cointegration test', 'Model ' + str(i), (setting_combinations_lst[i])[10])
		models_df.set_value('Include constant', 'Model ' + str(i), (setting_combinations_lst[i])[11])

		
	overview_df.set_value('Number of models', ' ', num_models)
	
	num_windows=[]
	for i in range(0,len(window_size_lst)):	
		num_windows.append(int((float(obs_per_asset)/float(window_size_lst[i])))-1)
	overview_df.set_value('Number of analysis windows (per model)', ' ', num_windows)
	overview_df.set_value('Number of trading windows (per model)', ' ', num_windows)	
	
	print
	print overview_df[-3:]
	
	# Generating the master dataframe template with the pairs as columns
	pair_list=[]
	lst = range(0, assets_df.shape[1])
	# Assigns each asset a number from 0 to len(assets) and generates tuples (0,1) (0,2) (0,3) etc representing each unique pair
	# Tuple (0,1) would represent the unique pair (asset1, asset2)
	for pair in itertools.combinations(lst, 2):
		# Looks up the asset name corresponding to the first number in the tuple
		name_y = assets_df.columns[pair[0]]
		# Looks up the asset name corresponding to the second number in the tuple
		name_x = assets_df.columns[pair[1]]
		# Joins the two names in one string
		pair_name = name_y + " & " + name_x
		pair_list.append(pair_name)
	dataframe_template_timeunit_pairs=pd.DataFrame(data=None, columns=pair_list, index=assets_df.index)
	all_models_profit_df = pd.DataFrame(data=None)
	all_models_num_trades_df = pd.DataFrame(data=None)
	all_models_profit_fd_df = pd.DataFrame(data=None)
	all_models_gross_return_df = pd.DataFrame(data=None, dtype=float)
	all_models_gross_return_fd_df = pd.DataFrame(data=None, dtype=float)

	# model_df_dict is used to store data
	# Key is the model number
	# Values are setting_combinations_lst, adf_summary, coint_summary, trade_summary
	model_df_dict = {}   
	print
	print
	with tqdm(total=num_models) as pbar:
		for i in range(0,num_models):
			loopstart = time.clock()
			model_df_dict[i] = roll(assets_df, overview_df, setting_combinations_lst[i], i, dataframe_template_timeunit_pairs)
			calc(model_df_dict, i, models_df, setting_combinations_lst, overview_df, all_models_profit_df, HHMM, all_models_profit_fd_df, assets_df, loopstart, all_models_num_trades_df)
			if save == "Y" and safe_save == "Y":
					save_function(model_df_dict, num_models, models_df, setting_combinations_lst, overview_df, all_models_profit_df, HHMM, all_models_profit_fd_df, assets_df, i=i)
			pbar.update(1)
	
	print
	if models_df.shape[1] <= 0:
		print models_df.iloc[-19:,].round(2)
	else:
		sorted_models_df=models_df.iloc[-19:,]
		best_worst_models_df = sorted_models_df.sort_values('Profit (USD)', axis=1, ascending=False).iloc[:,[0,-1]]
		best_title = "Best: "+str(best_worst_models_df.columns[0])
		worst_title = "Worst: "+str(best_worst_models_df.columns[1])
		best_worst_models_df.columns=[best_title, worst_title]
		print best_worst_models_df
	print


	all_models_profit_df = all_models_profit_df.dropna(how='all')
	all_models_num_trades_df = all_models_num_trades_df.replace(0,np.nan).dropna(how='all')

	for i in range(0,all_models_profit_df.shape[1]):
		for j in range(all_models_profit_df.index[0], all_models_profit_df.index[-1]+1):
			if all_models_num_trades_df.loc[j, 'model.'+str(i)] != 0:
				gross_return = (float(all_models_profit_df.loc[j, 'model.'+str(i)])/float(all_models_num_trades_df.loc[j, 'model.'+str(i)]))*100
				all_models_gross_return_df.set_value(j, 'model.'+str(i), gross_return)
	
	for i in range(0,all_models_profit_df.shape[1]):
		all_models_gross_return_fd_df['model.'+str(i)]= all_models_gross_return_df['model.'+str(i)] - all_models_gross_return_df['model.'+str(i)].shift()
	
	
	all_models_gross_return_fd_df_dropna = all_models_gross_return_fd_df.dropna(how='all')
	all_models_gross_return_fd_df_dropna = all_models_gross_return_fd_df_dropna.fillna(value=0)
	
	all_models_profit_fd_df_dropna = all_models_profit_fd_df.dropna(how='all')
	all_models_profit_fd_df_dropna = all_models_profit_fd_df_dropna.fillna(value=0)
	
	##############################################################
	
	### Generating a fake model to test StepM. NOT TO BE KEPT. ###
	#for i in range(0,20):
	#	shim_df = shim(all_models_profit_fd_df)
	#	i=all_models_profit_fd_df.shape[1]
	#	all_models_profit_fd_df['model.'+str(i)]=shim_df
	
	##############################################################
	
	
	
	
	stepm_results = multiple_testing(all_models_profit_fd_df_dropna, HHMM)
	
	time_elapsed = time.clock() - start
	time_elapsed = "%.2f" % (time_elapsed)
	
	overview_df.set_value('Best model', ' ', stepm_results[2].index.values[0])
	overview_df.set_value('StepM P-value of best model', ' ', stepm_results[2].iloc[0,1])
	overview_df.set_value('SPA P-value', ' ', stepm_results[2].iloc[0,2])
	
	
	if save == 'Y':
		save_function(model_df_dict, num_models, models_df, setting_combinations_lst, overview_df, all_models_profit_df, HHMM, all_models_profit_fd_df, assets_df, all_models_gross_return_df, stepm_results=stepm_results, i=num_models)
	
	print
	print "_________________________________________________________"
	print
	print "Code runtime: %s seconds." % (time_elapsed)
	
	ttsWatson.play("Analysis completed")




def roll(assets_df, overview_df, setting_combinations_lst, i, dataframe_template_timeunit_pairs):
	model=i
	window_size=setting_combinations_lst[3]
	obs_per_asset=overview_df.get_value('Number of observations (per asset)', ' ')
	num_windows=int((float(obs_per_asset)/float(window_size)))
	corr_prescreen = setting_combinations_lst[9]
	window_start_index = 0
	window_end_index = window_size-1
	i=0
	
	# Since window size may change from model to model, these templates need to be generated from scratch each time
	dataframe_template_window_pairs = pd.DataFrame(data=None, columns=dataframe_template_timeunit_pairs.columns, index=dataframe_template_timeunit_pairs[:num_windows].index)
	dataframe_template_window_assets = pd.DataFrame(data=None, columns=assets_df.columns, index=range(0,num_windows))
	all_adf_tests_df = pd.DataFrame(data=None, columns=dataframe_template_window_assets.columns, index=dataframe_template_window_assets.index)
	all_corr_tests_df = pd.DataFrame(data=None, columns=dataframe_template_window_pairs.columns, index=dataframe_template_window_pairs.index)
	all_coint_tests_df = pd.DataFrame(data=None, columns=dataframe_template_window_pairs.columns, index=dataframe_template_window_pairs.index)
	all_pairs_timeunit_df = pd.DataFrame(data=None, columns=dataframe_template_timeunit_pairs.columns, index=dataframe_template_timeunit_pairs.index)
	num_trades_timeunit_df = pd.DataFrame(data=0, columns=dataframe_template_timeunit_pairs.columns, index=dataframe_template_timeunit_pairs.index)
	all_trades_df = pd.DataFrame(data=None, columns=dataframe_template_timeunit_pairs.columns, index=dataframe_template_timeunit_pairs.index)
	diseq_df = pd.DataFrame(data=None, columns=dataframe_template_timeunit_pairs.columns, index=dataframe_template_timeunit_pairs.index)
	coint_for_trading_df=pd.DataFrame(data=None, index=None, columns=['Window','T-value'])
	corr_for_coint_df=pd.DataFrame(data=None, index=None, columns=['Window','P-value'])
	trade_dict={}
	
	while i < num_windows-1:	
		window_start_index=(window_size)*i
		window_end_index=window_start_index+window_size-1
		analysis_window_df = assets_df.loc[window_start_index:window_end_index]
		trade_window_df = assets_df.loc[window_end_index+1:window_end_index+window_size]
		# Finds columns which are missing values for this window so that they can be skipped
		incomplete_columns_lst = analysis_window_df.columns[analysis_window_df.isnull().any()].tolist()
		if i < num_windows-1:
			# ADF tests
			adf_summary = adf_test(setting_combinations_lst, analysis_window_df, i, all_adf_tests_df, incomplete_columns_lst)
			adf_failures_dict = adf_summary[1]
			# Correlation screening
			if corr_prescreen == 'Y':
				corr_summary = corr(setting_combinations_lst, analysis_window_df, i, all_corr_tests_df, corr_for_coint_df, incomplete_columns_lst)
				corr_for_coint_df = corr_summary[1]
			if corr_prescreen == 'N':
				corr_summary = (pd.DataFrame(data=None), pd.DataFrame(data=None))
			# Cointegration tests
			coint_summary = coint(setting_combinations_lst, analysis_window_df, i, all_coint_tests_df, coint_for_trading_df, adf_failures_dict, overview_df, corr_for_coint_df, incomplete_columns_lst)
			coint_for_trading_df = coint_summary[0]
			# Trading
			trade_summary = trade(setting_combinations_lst, analysis_window_df, trade_window_df, i, all_pairs_timeunit_df, all_trades_df, diseq_df, coint_for_trading_df, trade_dict, model, num_trades_timeunit_df)
			i+=1
	
	return (setting_combinations_lst, adf_summary, coint_summary, trade_summary, corr_summary)





def adf_test(setting_combinations_lst, analysis_window_df, i, all_adf_tests_df, incomplete_columns_lst):
	window = i
	incl_constant=setting_combinations_lst[11]
	num_assets = analysis_window_df.shape[1]
	adf_failures_dict = {}
	adf_failures_dict.clear()
	if incl_constant == 'Y':
		specification = 'c'
	if incl_constant == 'N':
		specification = 'nc'

	# Description:
		# The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root, with the alternative that there is no unit root. 
		# If the pvalue is above a critical size, then we cannot reject that there is a unit root.
	# Usage:
		# ts.adfuller(x, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
	# Parameters:	
		#x - (array_like, 1d) data series
		# maxlag - Maximum lag which is included in test, default 12*(nobs/100)^{1/4}
		# regression - {‘c’,’ct’,’ctt’,’nc’} Constant and trend order to include in regression
			# ‘c’ : constant only (default)
			# ‘ct’ : constant and trend
			# ‘ctt’ : constant, and linear and quadratic trend
			# ‘nc’ : no constant, no trend
		# autolag : {‘AIC’, ‘BIC’, ‘t-stat’, None}
			# if None, then maxlag lags are used
			# if ‘AIC’ (default) or ‘BIC’, then the number of lags is chosen to minimize the corresponding information criterion
	
	for i in range(0, num_assets):
		name = analysis_window_df.columns[i]
		if name not in incomplete_columns_lst:
			adf_p_val = (ts.adfuller(analysis_window_df.iloc[:,i].values, maxlag=None, regression=specification, autolag='AIC', store=False, regresults=False))[1]
			all_adf_tests_df.set_value(window, name, adf_p_val)
	
	
	assets_form_fd_df = analysis_window_df.shift() - analysis_window_df
	

	for i in range(0, num_assets):
		name = assets_form_fd_df.columns[i]

		if name not in incomplete_columns_lst:
			
			adf_p_val = (ts.adfuller(assets_form_fd_df.iloc[1:,i].values, maxlag=None, regression=specification, autolag='AIC', store=False, regresults=False))[1]
			all_adf_tests_df.set_value(window, "FD "+ name, adf_p_val)
		if name in incomplete_columns_lst:
			all_adf_tests_df.set_value(window, "FD "+ name, np.nan)
	
	user_p_val=setting_combinations_lst[0]
	for i in range(0, num_assets):
		name = analysis_window_df.columns[i]
		if all_adf_tests_df.iloc[window][name]  < user_p_val:
			adf_failures_dict[name] = 'Stationary in levels'
		if all_adf_tests_df.iloc[window]["FD "+ name]  > user_p_val:
			adf_failures_dict[name] = 'Not stationary in FD'

	
	return (all_adf_tests_df, adf_failures_dict)





def corr(setting_combinations_lst, analysis_window_df, i, all_corr_tests_df, corr_for_coint_df, incomplete_columns_lst):
	window = i
	user_p_val = setting_combinations_lst[1]
	for pair in all_corr_tests_df.columns:
			asset1, asset2 = pair.split(" & ")
			if asset1 not in incomplete_columns_lst and asset2 not in incomplete_columns_lst:
				y = analysis_window_df.loc[:, asset1]
				x = analysis_window_df.loc[:, asset2]
				results = pearsonr(y, x)
				p_corr = results[0]
				corr_p_val = results[1]
				all_corr_tests_df.set_value(window, pair, corr_p_val)
				if corr_p_val < user_p_val:
					dex = corr_for_coint_df.shape[0]
					corr_for_coint_df.set_value(dex, 'Window', window)
					corr_for_coint_df.set_value(dex, 'Pair', pair)
					corr_for_coint_df.set_value(dex, 'P-value', corr_p_val)
					corr_for_coint_df.set_value(dex, 'Correlation', p_corr)
	corr_for_coint_df = corr_for_coint_df.sort_values(by=['Window','P-value'])
	
	return(all_corr_tests_df, corr_for_coint_df)




def coint(setting_combinations_lst, analysis_window_df, i, all_coint_tests_df, coint_for_trading_df, adf_failures_dict, overview_df, corr_for_coint_df, incomplete_columns_lst):
	window = i
	user_p_val=setting_combinations_lst[2]
	drop_adf=setting_combinations_lst[8]
	coint_test=setting_combinations_lst[10]
	incl_constant=setting_combinations_lst[11]
	corr_prescreen=setting_combinations_lst[9]
	
	corr_for_coint_current_window_df = pd.DataFrame(data=None)
	corr_for_coint_current_window_df = corr_for_coint_df.loc[corr_for_coint_df['Window'] == window]
	
	eligible_pairs = []
	del eligible_pairs[:]
	
	if corr_prescreen == 'Y' and drop_adf == 'Y':
		#print "Yes corr prescreen and yes drop adf"
		for pair in corr_for_coint_current_window_df.loc[:,'Pair']:
			asset1, asset2 = pair.split(" & ")
			if asset1 not in incomplete_columns_lst and asset2 not in incomplete_columns_lst:
				if asset1 not in adf_failures_dict.keys() and asset2 not in adf_failures_dict.keys():
					eligible_pairs.append(pair)
		
	if corr_prescreen == 'Y' and drop_adf == 'N':
		#print "Yes orr prescreen but no drop adf"
		for pair in corr_for_coint_current_window_df.loc[:,'Pair']:
			asset1, asset2 = pair.split(" & ")
			if asset1 not in incomplete_columns_lst and asset2 not in incomplete_columns_lst:
				eligible_pairs.append(pair)
				
	if corr_prescreen == 'N' and drop_adf == 'Y':
		#print "No corr prescreen but yes drop adf"
		for pair in all_coint_tests_df.columns:
			asset1, asset2 = pair.split(" & ")
			if asset1 not in incomplete_columns_lst and asset2 not in incomplete_columns_lst:
				if asset1 not in adf_failures_dict.keys() and asset2 not in adf_failures_dict.keys():
					eligible_pairs.append(pair)
	
	if corr_prescreen == 'N' and drop_adf == 'N':
		#print "No corr prescreen and no drop adf"
		for pair in all_coint_tests_df.columns:
			asset1, asset2 = pair.split(" & ")
			if asset1 not in incomplete_columns_lst and asset2 not in incomplete_columns_lst:
				eligible_pairs.append(pair)
	
	for pair in eligible_pairs:
		asset1, asset2 = pair.split(" & ")
		if asset1 not in adf_failures_dict.keys() and asset2 not in adf_failures_dict.keys():
			y = analysis_window_df.loc[:, asset1]
			x = analysis_window_df.loc[:, asset2]
			pair_cointegrated = False
			
			if coint_test == 'PO':
				# Usage:
					# ca.po(z, demean = c("none", "constant", "trend"), lag = c("short", "long"), type = c("Pu", "Pz"), tol = NULL)
				# Arguments:
					# z - Data matrix to be investigated for cointegration.
					# demean - The method for detrending the series, either "none", "constant" or "trend".
					# lag - Either a short or long lag number used for variance/covariance correction.
					# type - The test type, either "Pu" or "Pz".
					# tol - Numeric, this argument is passed to solve() in ca.po().
				ro.globalenv['y']=ro.FloatVector(y)
				ro.globalenv['x']=ro.FloatVector(x)
				r('library(urca)')
				if incl_constant == 'Y':
					r('results <- (ca.po(cbind(y,x), demean = "constant"))')
				if incl_constant == 'N':
					r('results <- (ca.po(cbind(y,x), demean = "none"))')
				coint_tval = r('results@teststat')[0]
				coint_tval = abs(coint_tval)
				po_cvals = r('cVals <- results@cval')
				
				if user_p_val == 0.01:
					selector = 2
				if user_p_val == 0.05:
					selector = 1
				if user_p_val == 0.10:
					selector = 0
				if coint_tval > po_cvals[selector]:
					pair_cointegrated = True
			
			if coint_test == 'EG':
				#Usage:
					# ts.coint(y0, y1, trend='c', method='aeg', maxlag=None, autolag='aic', return_results=None)
				#Arguements:
					# y1 - first element in cointegrating vector
					# y2 - remaining elements in cointegrating vector
					# trend - trend term included in regression for cointegrating equation * ‘c’ : constant * ‘ct’ : constant and linear trend * also available quadratic trend ‘ctt’, and no constant ‘nc’
					# method - currently only ‘aeg’ for augmented Engle-Granger test is available. default might change.
					# maxlag - (None or int) keyword for adfuller, largest or given number of lags
					# autolag - keyword for adfuller, lag selection criterion.
				if incl_constant == 'Y':
					specification = 'c'
				if incl_constant == 'N':
					specification = 'nc'
				results = ts.coint(y, x, trend=specification)
				coint_tval = results[0]
				coint_tval = abs(coint_tval)
				coint_pval = results[1]
				if float(coint_pval) < user_p_val:
					pair_cointegrated = True
			
			# To save processing time, the cointegrating relation is ony computed for pairs who have been found to be cointegrated
			if pair_cointegrated == False:
				coint_relation = ' '
				constant = ' '
				
			if pair_cointegrated == True:
				if incl_constant == 'Y':
					x = ts.add_constant(x)
					model = ts.OLS(y, x)
					reg = model.fit()
					coint_relation=float((reg.params)[1])
					constant=float((reg.params)[0])
				if incl_constant == 'N':
					model = ts.OLS(y, x)
					reg = model.fit()
					coint_relation=float((reg.params)[0])
					constant=0
				#sys.exit()
				# In addition to the large dataframe cointaining all results, a smaller one is also generated which contains just the results of the pairs elegible for trading
				# Each (window, pair) cell in the dataframe is a list with (up to) three elements; the p-value, cointegrating relation and the constant
				dex = coint_for_trading_df.shape[0]
				coint_for_trading_df.set_value(dex, 'Window' , window)
				coint_for_trading_df.set_value(dex, 'Pair' , pair)
				coint_for_trading_df.set_value(dex, 'T-value', coint_tval)
				coint_for_trading_df.set_value(dex, 'Cointegrating relation' , coint_relation)
				coint_for_trading_df.set_value(dex, 'Constant' , constant)
			all_coint_tests_df.set_value(window, pair, coint_tval)
	
	coint_for_trading_df = coint_for_trading_df.sort_values(by=['Window','T-value'], ascending=False)
	return (coint_for_trading_df, all_coint_tests_df)





def trade(setting_combinations_lst, analysis_window_df, trade_window_df, i, all_pairs_timeunit_df, all_trades_df, diseq_df, coint_for_trading_df, trade_dict, model, num_trades_timeunit_df):

	window = i
	open_trigger_sensitivity = setting_combinations_lst[4]
	close_trigger_sensitivity = setting_combinations_lst[5]
	stoploss_trigger_sensitivity = setting_combinations_lst[6]
	asymmetric_legs = setting_combinations_lst[7]
	stop_loss_type = setting_combinations_lst[12]
	coint_for_trading_current_window_df = pd.DataFrame(data=None)
	coint_for_trading_current_window_df = coint_for_trading_df.loc[coint_for_trading_df['Window'] == window]
	
	for i in range(0, coint_for_trading_df.shape[0]):
		pair = coint_for_trading_df.iloc[i]['Pair']
		# The below condition finds pairs have been traded previously, but are not elegible for trading in this window
		# The 'all_pairs_timeunit_df' value for pairs which meet this condition can then be set appropriately for all cells in this window
		j = trade_window_df.index[0]-1
		if pair not in str(coint_for_trading_current_window_df['Pair']):
			for t in range(trade_window_df.index[0], trade_window_df.index[-1]+1):
				all_pairs_timeunit_df.set_value(t, pair, all_pairs_timeunit_df.loc[j, pair])
	
	
	# We trade one pair at a time per window 
	for i in range(0, coint_for_trading_current_window_df.shape[0]):
		trades = 0
		pair = coint_for_trading_current_window_df.iloc[i]['Pair']
		coint_relation = coint_for_trading_df.iloc[i]['Cointegrating relation']
		constant = coint_for_trading_df.iloc[i]['Constant']
		if asymmetric_legs == 'N':
			position_size_a1 = 1
			position_size_a2 = 1
		if asymmetric_legs == 'Y':
			position_size_a1 = (2-2*(1/coint_relation))
			position_size_a2 = (2*(1/coint_relation))
		# We need to recover the original price series from the analysis and trade dataframes.
		# To do this we manualy seperate the names of each asset in the pair and use this for refrencing
		asset1, asset2 = pair.split(" & ")

		# The trigger for each pair is calculated using the analysis (formation period) window
		# The higher trigger_sensitivity, the less sensitive the trigger is
		# Trigger will always be positive since it is based on the standard deviation of the analysis window disequlibrium
		open_trigger = open_trigger_sensitivity*np.std(analysis_window_df.loc[:, asset1] - (constant + coint_relation*analysis_window_df.loc[:, asset2]))
		close_trigger = close_trigger_sensitivity*np.std(analysis_window_df.loc[:, asset1] - (constant + coint_relation*analysis_window_df.loc[:, asset2])) 
		
		# Calculate the historical standard deviation of the net value of a pair trade and multiply by the desired trigger sensitivity
		if stop_loss_type == 'P' or stop_loss_type == 'PD':
			stoploss_trigger = stoploss_trigger_sensitivity*np.std(analysis_window_df.loc[:, asset1] - (coint_relation*analysis_window_df.loc[:, asset2]))
		# Having no stop loss is implemented by setting the trigger impossibly high
		if stop_loss_type == 'N':
			stoploss_trigger = 100**10
		
		# Disequilibrium is calculated using the trading window 
		for t in range(trade_window_df.index[0], trade_window_df.index[-1]+1):
			diseq_df.set_value(t, pair, trade_window_df.loc[t, asset1] - (constant + coint_relation*trade_window_df.loc[t, asset2]))
		stop_loss='N'
		position_open=False
		
		
		
		
		# Once the stop loss is triggered all trading in that pair is ceased for the rest of the window
		for t in range(trade_window_df.index[0], trade_window_df.index[-1]+1):
				j = trade_window_df.index[0]-1
				diseq = diseq_df.loc[t,pair]
				p1_c = trade_window_df.loc[t, asset1] 
				p2_c = trade_window_df.loc[t, asset2]
				diseq_c = diseq
		
				
				# stop_loss_bool under the PD setting will prevent a position from being opened if the estimated disequlibrium is too large
				if stop_loss_type == 'PD':
					stop_loss_bool = abs(diseq) >= stoploss_trigger
				
				# Under all other settings, no constraints are placed on opening a position when disequlibrium is high (stop_loss_bool is always false)
				if stop_loss_type == 'P' or stop_loss_type == 'N':
					stop_loss_bool = False
			
					
				if position_open == False:
					# If no position is open and the trigger condition is not met, do nothing
					if abs(diseq) <= open_trigger or stop_loss_bool == True or stop_loss=='Y':
						# The pair + "trade profit" column only gets created once that asset has been traded
						# The below is equivalent to "if pair traded before"
						if pair + " Trade Profit" in all_trades_df.columns:
							all_pairs_timeunit_df.set_value(t, pair, all_trades_df.loc[0:t-1,pair + " Trade Profit"].sum())
					# Opens a position for a given pair if the trigger condition is met, no position for that pair is open yet and it is not the last time period in the window
					if abs(diseq) > open_trigger and stop_loss_bool == False and not t == trade_window_df.index[-1] and stop_loss=='N':
						p1_o = p1_c
						p2_o = p2_c
						diseq_o = diseq
						k=t
						all_trades_df.set_value(t, pair + " p1_o", p1_o)
						all_trades_df.set_value(t, pair + " p2_o", p2_o)
						if diseq < 0:
							a1="L"
							all_trades_df.set_value(t, pair + " Blurb", "Position opened: " + asset1 + " long & " + asset2 + " short @ t=" + str(t))
							all_trades_df.set_value(t, pair + " Disequilibrium", diseq_o)
							leg1 = position_size_a1-(p1_o*position_size_a1)/p1_c
							leg2 = -(position_size_a2-(p2_o*position_size_a2)/p2_c)
							marktomkt = leg1 + leg2
							# If this pair has been traded before
							if pair + " Trade Profit" in all_trades_df.columns:
								marktomkt+=all_trades_df.loc[0:t-1,pair + " Trade Profit"].sum()
							if not pair + " Trade Profit" in all_trades_df.columns:
								all_trades_df[pair + " Trade Profit"] = np.NaN
							all_pairs_timeunit_df.set_value(t, pair, marktomkt)
							trades+=2
							num_trades_timeunit_df.set_value(t, pair, trades)
						if diseq > 0:
							a1="S"
							all_trades_df.set_value(t, pair + " Blurb", "Position opened: " + asset1 + " short & " + asset2 + " long @ t=" + str(t))
							all_trades_df.set_value(t, pair + " Disequilibrium", diseq_o)
							leg1 = -(position_size_a1-(p1_o*position_size_a1)/p1_c)
							leg2 = (position_size_a2-(p2_o*position_size_a2)/p2_c)
							marktomkt = leg1 + leg2
							if pair + " Trade Profit" in all_trades_df.columns:
								marktomkt+=all_trades_df.loc[0:t-1,pair + " Trade Profit"].sum()
							if not pair + " Trade Profit" in all_trades_df.columns:
								all_trades_df[pair + " Trade Profit"] = np.NaN
							all_pairs_timeunit_df.set_value(t, pair, marktomkt)
							trades+=2
							num_trades_timeunit_df.set_value(t, pair, trades)
						position_open = True
				
				
				elif position_open == True:
					
					if a1 == "L":
						leg1 = position_size_a1-(p1_o*position_size_a1)/p1_c
						leg2 = -(position_size_a2-(p2_o*position_size_a2)/p2_c)
					if a1 == "S":
						leg1 = -(position_size_a1-(p1_o*position_size_a1)/p1_c)
						leg2 = position_size_a2-(p2_o*position_size_a2)/p2_c
					marktomkt = leg1 + leg2
				
					# Close the trade (convergence)
					if diseq_o*diseq_c<0:
						position_open = False
						close_code = 1
						close_trade(a1, asset1, asset2, t, p1_o, p2_o, p1_c, p2_c, close_code, all_trades_df, all_pairs_timeunit_df, pair, diseq_o, diseq_c, position_size_a1, position_size_a2, k, trade_dict)
					if abs(diseq) <= close_trigger:
						close_code = 1
						close_trade(a1, asset1, asset2, t, p1_o, p2_o, p1_c, p2_c, close_code, all_trades_df, all_pairs_timeunit_df, pair, diseq_o, diseq_c, position_size_a1, position_size_a2, k, trade_dict)
						position_open = False
						
					# Close the trade (profit stop loss)
					if stop_loss_type == 'P' or stop_loss_type == 'PD':
						if marktomkt <= -stoploss_trigger:
							position_open = False
							close_code = 2
							close_trade(a1, asset1, asset2, t, p1_o, p2_o, p1_c, p2_c, close_code, all_trades_df, all_pairs_timeunit_df, pair, diseq_o, diseq_c, position_size_a1, position_size_a2, k, trade_dict)
							stop_loss='Y'
						
					# Close the trade (end of window)
					if t == trade_window_df.index[-1]:
						position_open = False
						close_code = 3
						close_trade(a1, asset1, asset2, t, p1_o, p2_o, p1_c, p2_c, close_code, all_trades_df, all_pairs_timeunit_df, pair, diseq_o, diseq_c, position_size_a1, position_size_a2, k, trade_dict)
						
					
					# Trade remains open
					elif stop_loss_type == 'P' or stop_loss_type == 'PD':
						if diseq_o*diseq_c>0 and marktomkt > -stoploss_trigger:
							marktomkt+=all_trades_df.loc[0:t-1,pair + " Trade Profit"].sum()
							all_pairs_timeunit_df.set_value(t, pair, marktomkt)
					elif stop_loss_type == 'N':
						if diseq_o*diseq_c>0:
							marktomkt+=all_trades_df.loc[0:t-1,pair + " Trade Profit"].sum()
							all_pairs_timeunit_df.set_value(t, pair, marktomkt)
		
		
		# Graphing disequlibrium (for troubleshooting - have this off for normal operation)
		plt_diseq_df=pd.DataFrame(data=None)
		plt_diseq_df['Diseq']=diseq_df.loc[:,pair].dropna(how='all')
		plt_diseq_df['+Open']=open_trigger
		plt_diseq_df['-Open']=-open_trigger
		plt_diseq_df['+Close']=close_trigger
		plt_diseq_df['-Close']=-close_trigger
		plt_diseq_df['Trade O']=np.nan
		plt_diseq_df['Trade C c']=np.nan
		plt_diseq_df['Trade C w']=np.nan
		plt_diseq_df['Trade C s']=np.nan
		plt_diseq_df['Trade O*']=np.nan
		plt_diseq_df['Trade C c*']=np.nan
		plt_diseq_df['Trade C w*']=np.nan
		plt_diseq_df['Trade C s*']=np.nan
		plt_diseq_df['a1']=trade_window_df.loc[:,asset1]
		plt_diseq_df['a2']=trade_window_df.loc[:,asset2]
		plt_diseq_df['Profit']=all_pairs_timeunit_df.loc[:,pair].fillna(0)
		
		
		if str(pair) + " p1_o" in all_trades_df.columns:
			lst = all_trades_df.loc[:,str(pair) + " p1_o"].dropna().index.values.tolist()
			print
			print len(lst)
			for t in lst:
					plt_diseq_df.set_value(t,'Trade O',diseq_df.loc[t,pair])
					plt_diseq_df.set_value(t,'Trade O*',plt_diseq_df.loc[t,'Profit'])
					
		if str(pair) + " Close Code" in all_trades_df.columns:
			for t in range(trade_window_df.index[0], trade_window_df.index[-1]+1):
				if all_trades_df.loc[t,str(pair) + " Close Code"] == 1:
					plt_diseq_df.set_value(t,'Trade C c',diseq_df.loc[t,pair])
					plt_diseq_df.set_value(t,'Trade C c*',plt_diseq_df.loc[t,'Profit'])
				if all_trades_df.loc[t,str(pair) + " Close Code"] == 2:
					plt_diseq_df.set_value(t,'Trade C s',diseq_df.loc[t,pair])
					plt_diseq_df.set_value(t,'Trade C s*',plt_diseq_df.loc[t,'Profit'])
				if all_trades_df.loc[t,str(pair) + " Close Code"] == 3:
					plt_diseq_df.set_value(t,'Trade C w',diseq_df.loc[t,pair])
					plt_diseq_df.set_value(t,'Trade C w*',plt_diseq_df.loc[t,'Profit'])


		fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
		
		l1, = ax1.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'Diseq'].values, color=sns.xkcd_rgb["medium blue"])
		l2, = ax1.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'+Open'].values, color=sns.xkcd_rgb["grey"], linestyle=':', label='Open Trigger')
		l3, = ax1.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'-Open'].values, color=sns.xkcd_rgb["grey"], linestyle=':')
		l4, = ax1.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'+Close'].values, color=sns.xkcd_rgb["black"], linestyle=':', label='Close Trigger')
		l5, = ax1.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'-Close'].values, color=sns.xkcd_rgb["black"], linestyle=':')
		l6, = ax1.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'Trade O'].values, color=sns.xkcd_rgb["grey"], marker='X', markersize=10, label='Position Opened')
		l7, = ax1.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'Trade C c'].values, color='k', marker='X', markersize=10, label='Trade Closed (Convergence)')
		l8, = ax1.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'Trade C w'].values, color=sns.xkcd_rgb["greenish"], marker='X', markersize=10, label='Trade Closed (End of Window)')
		l9, = ax1.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'Trade C s'].values, color=sns.xkcd_rgb["red orange"], marker='X', markersize=10, label='Trade Closed (Stop Loss)')
		ax1.set_ylabel('Disequlibrium', size=22)	
		#ax1.get_yaxis().set_label_coords(-0.05,0.5)
		
		ax2.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'Profit'].values, color=sns.xkcd_rgb["medium blue"], label='Profit')
		ax2.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'Trade O*'].values, color=sns.xkcd_rgb["grey"], marker='X', markersize=10)
		ax2.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'Trade C c*'].values, color='k', marker='X', markersize=10)
		ax2.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'Trade C w*'].values, color=sns.xkcd_rgb["greenish"], marker='X', markersize=10)
		ax2.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'Trade C s*'].values, color=sns.xkcd_rgb["red orange"], marker='X', markersize=10)
		ax2.set_ylabel('Profit', size=22)
		ax2.set_xlabel('Days', size=22)
		#ax2.get_yaxis().set_label_coords(-0.05,0.5)
		
		ax1.legend(ncol=2,loc=0, fontsize=14)
		
		#ax3.set_ylabel('Exchange rates')
		#ax3.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'a1'].values, sns.xkcd_rgb["medium blue"])
		#ax3.plot(plt_diseq_df.index.values, plt_diseq_df.loc[:,'a2'].values, sns.xkcd_rgb["grey"])
		#ax3.get_yaxis().set_label_coords(-0.05,0.5)
		#plt.suptitle(pair)
		plt.xlim(plt_diseq_df.index.values[0],plt_diseq_df.index.values[-1])
		fig.tight_layout() 
		fig.subplots_adjust(top=0.90)
		ax1.tick_params(axis='both', which='major', labelsize=14)
		ax2.tick_params(axis='both', which='major', labelsize=14)
		plt.savefig(pair+' m'+ str(model)+' w'+ str(window)+'.png', bbox_inches='tight')
		plt.show()

	return(all_pairs_timeunit_df, all_trades_df, diseq_df, trade_dict, num_trades_timeunit_df)





def close_trade(a1, asset1, asset2, t, p1_o, p2_o, p1_c, p2_c, close_code, all_trades_df, all_pairs_timeunit_df, pair, diseq_o, diseq_c, position_size_a1, position_size_a2, k, trade_dict):
	
	if close_code == 1:
		if a1 == "L":
			all_trades_df.set_value(t, pair + " Blurb", "Position closed (convergence): " + asset1 + " short & " + asset2 + " long @ t=" + str(t))
		if a1 == "S": 
			all_trades_df.set_value(t, pair + " Blurb", "Position closed (convergence): " + asset1 + " long & " + asset2 + " short @ t=" + str(t))
	if close_code == 2:
		if a1 == "L":
			all_trades_df.set_value(t, pair + " Blurb", "Position closed (stop loss): " + asset1 + " short & " + asset2 + " long @ t=" + str(t))
		if a1 == "S": 
			all_trades_df.set_value(t, pair + " Blurb", "Position closed (stop loss): " + asset1 + " long & " + asset2 + " short @ t=" + str(t))
	if close_code == 3:
		if a1 == "L":
			all_trades_df.set_value(t, pair + " Blurb", "Position closed (end of window): " + asset1 + " short & " + asset2 + " long @ t=" + str(t))
		if a1 == "S": 
			all_trades_df.set_value(t, pair + " Blurb", "Position closed (end of window): " + asset1 + " long & " + asset2 + " short @ t=" + str(t))
	
	duration = t-k+1
	all_trades_df.set_value(t, pair + " Open", k)
	all_trades_df.set_value(t, pair + " Duration", duration)
	all_trades_df.set_value(t, pair + " p1", p1_c)
	all_trades_df.set_value(t, pair + " p2", p2_c)
	all_trades_df.set_value(t, pair + " Disequilibrium", diseq_c)
	all_trades_df.set_value(t, pair + " Close Code", close_code)

	if a1 == "L":
		leg1 = position_size_a1-(p1_o*position_size_a1)/p1_c
		leg2 = -(position_size_a2-(p2_o*position_size_a2)/p2_c)
	if a1 == "S":
		leg1 = -(position_size_a1-(p1_o*position_size_a1)/p1_c)
		leg2 = position_size_a2-(p2_o*position_size_a2)/p2_c
		
	marktomkt = leg1 + leg2
	marktomkt_cu = marktomkt + all_trades_df.loc[0:t-1,pair + " Trade Profit"].sum()
	all_trades_df.set_value(t, pair + " Leg 1", leg1)
	all_trades_df.set_value(t, pair + " Leg 2", leg2)
	all_trades_df.set_value(t, pair + " Trade Profit", marktomkt)
	all_pairs_timeunit_df.set_value(t, pair, marktomkt_cu)
	
	if marktomkt == 0:
		print
		print
		print "Warning: Attempting to close a position when mark-to-market is 0"
		print
		print "Leg 1 (%s): %s" % (asset1, round(leg1,3))
		print "Leg 2 (%s): %s" % (asset2, round(leg2,3))
		print "Duration: %s" % (duration)
		print
		preformance_code=5
	if marktomkt > 0 and abs(diseq_o)>abs(diseq_c):
		preformance_code=1
	if marktomkt > 0 and abs(diseq_o)<abs(diseq_c):
		preformance_code=2
	if marktomkt < 0 and abs(diseq_o)>abs(diseq_c):
		preformance_code=3
	if marktomkt < 0 and abs(diseq_o)<abs(diseq_c):
		preformance_code=4
		
	i=len(trade_dict)
	trade_dict[i] = pair, marktomkt, k, duration, close_code, preformance_code
	all_trades_df.loc[t, pair + " Preformance Code"] = preformance_code
	return





def calc(model_df_dict, i, models_df, setting_combinations_lst, overview_df, all_models_profit_df, HHMM, all_models_profit_fd_df, assets_df, loopstart, all_models_num_trades_df):
	
	obs_per_asset=overview_df.loc['Number of observations (per asset)'][' ']

	num_windows=int(float(obs_per_asset)/float(setting_combinations_lst[i][3]))-1
	
	split = (model_df_dict[i][1][0].dropna(how='all', axis=1).shape[1])/2
	
	stationary_levels_avg = (model_df_dict[i][1][0].dropna(how='all', axis=1).iloc[:,:split] < setting_combinations_lst[i][0]).sum(axis=1).sum(axis=0)/float(num_windows)
	stationary_fd_avg = (model_df_dict[i][1][0].dropna(how='all', axis=1).iloc[:,split:] < setting_combinations_lst[i][0]).sum(axis=1).sum(axis=0)/float(num_windows)
	
	if not model_df_dict[i][4][1].empty:
		corr_avg = model_df_dict[i][4][1].shape[0] /float(num_windows)
		corr_avg = round(corr_avg, 2)
	if model_df_dict[i][4][1].empty:
		corr_avg = "N/A"
	

	coint_avg = model_df_dict[i][2][0].shape[0]/float(num_windows)
	
	profit = (model_df_dict[i][3][0]).iloc[-1,:].sum(axis=0)
	trades = model_df_dict[i][3][1].filter(regex='Blurb').count().sum()*2
	if trades != 0:
		profit = (model_df_dict[i][3][0]).iloc[-1,:].sum(axis=0)
	if trades == 0:
		profit = 0
	
	all_models_profit_df['model.'+str(i)] = (model_df_dict[i][3][0]).sum(axis=1)
	all_models_profit_fd_df['model.'+str(i)]= all_models_profit_df['model.'+str(i)] - all_models_profit_df['model.'+str(i)].shift()
	avg_daily_change = (all_models_profit_fd_df.loc[:,'model.'+str(i)]).mean()
	all_models_profit_fd_df = all_models_profit_fd_df.replace(0, np.NaN)
	avg_daily_change_excl_0 = (all_models_profit_fd_df.loc[:,'model.'+str(i)]).mean()
	
	
	num_trades_timeunit_df = (model_df_dict[i][3][4])
	num_trades_timeunit_df = num_trades_timeunit_df.replace(0,np.nan)
	num_trades_timeunit_df = num_trades_timeunit_df.fillna(method = 'ffill')
	all_models_num_trades_df['model.'+str(i)] = num_trades_timeunit_df.sum(axis=1)
	
	# Working out the number of successful trades
	# Note that each 'trade' contains four subtrades (leg 1 open, leg 2 open, leg 1 close, leg 2 close)
	all_trades_df = model_df_dict[i][3][1]
	all_trades_regex_pc = all_trades_df.filter(regex='Preformance Code').dropna(how='all')
	num_pc1 = (all_trades_regex_pc == 1).sum(axis=0).sum()
	num_pc2 = (all_trades_regex_pc == 2).sum(axis=0).sum()
	num_pc3 = (all_trades_regex_pc == 3).sum(axis=0).sum()
	num_pc4 = (all_trades_regex_pc == 4).sum(axis=0).sum()
	num_pc5 = (all_trades_regex_pc == 5).sum(axis=0).sum()
	total_pc = num_pc1 + num_pc2 + num_pc3 + num_pc4 + num_pc5
	num_prof_win = num_pc1 + num_pc2
	num_prof_loss = num_pc3 + num_pc4
	num_diseq_win = num_pc1 + num_pc3
	num_diseq_loss = num_pc2 + num_pc4
	
	# Close code 1: Closed because of convergence
	# Close code 2: Closed because of stop loss
	# Close code 3: Closed because end of window
	num_close_code_1 = (model_df_dict[i][3][1].filter(regex='Close Code').dropna(how='all') == 1).sum(axis=0).sum()
	num_close_code_2 = (model_df_dict[i][3][1].filter(regex='Close Code').dropna(how='all') == 2).sum(axis=0).sum()
	num_close_code_3 = (model_df_dict[i][3][1].filter(regex='Close Code').dropna(how='all') == 3).sum(axis=0).sum()
	
	if trades != 0:
		gross_return = float(profit)/float(trades)*2*100
		pct_prof_win = float(num_prof_win)/float(num_prof_win+num_prof_loss)
		pct_diseq_win = float(num_diseq_win)/float(num_diseq_win+num_diseq_loss)
		pct_pc1 = float(num_pc1)/float(total_pc)
		pct_pc2 = float(num_pc2)/float(total_pc)
		pct_pc3 = float(num_pc3)/float(total_pc)
		pct_pc4 = float(num_pc4)/float(total_pc)
		pct_pc5 = float(num_pc5)/float(total_pc)
		pct_close_code_1 = float(num_close_code_1)/float(num_close_code_1 + num_close_code_2 + num_close_code_3)
		pct_close_code_2 = float(num_close_code_2)/float(num_close_code_1 + num_close_code_2 + num_close_code_3)
		pct_close_code_3 = float(num_close_code_3)/float(num_close_code_1 + num_close_code_2 + num_close_code_3)
		avg_duration = (model_df_dict[i][3][1].filter(regex='Duration').dropna(how='all').sum().sum())/(model_df_dict[i][3][1].filter(regex='Duration').dropna(how='all')).count().sum()
	else:
		gross_return = 0
		pct_prof_win = 0
		pct_diseq_win = 0
		pct_pc1 = 0
		pct_pc2 = 0
		pct_pc3 = 0
		pct_pc4 = 0
		pct_pc5 = 0
		pct_close_code_1 = 0
		pct_close_code_2 = 0
		pct_close_code_3 = 0
		avg_duration = 0

	time_elapsed = time.clock() - start
	
	stationary_levels_avg = round(stationary_levels_avg, 2)
	stationary_fd_avg = round(stationary_fd_avg, 2)
	coint_avg = round(coint_avg, 2)
	profit = round(profit, 2)
	gross_return = round(gross_return, 2)
	pct_prof_win = round(pct_prof_win, 4)
	pct_diseq_win = round(pct_diseq_win, 4)
	avg_duration = round(avg_duration, 2)
	pct_pc1 = round(pct_pc1, 4)
	pct_pc2 = round(pct_pc2, 4)
	pct_pc3 = round(pct_pc3, 4)
	pct_pc4 = round(pct_pc4, 4)
	pct_pc5 = round(pct_pc5, 4)
	pct_close_code_1= round(pct_close_code_1, 4)
	pct_close_code_2 = round(pct_close_code_2, 4)
	pct_close_code_3 = round(pct_close_code_3, 4)
	avg_daily_change = round(avg_daily_change, 4)
	avg_daily_change_excl_0 = round(avg_daily_change_excl_0, 4)
	time_elapsed = round(time_elapsed, 4)
	
	
	if models_df.loc['Stop loss type', 'Model ' + str(i)] == 'N':
		models_df.loc['Stop loss trigger sensitivity', 'Model ' + str(i)] = 'N/A'
		
	overview_df.set_value('Code runtime (s)', ' ', time_elapsed)
	models_df.set_value('Avg # stationary series (levels)', 'Model ' + str(i), stationary_levels_avg)
	models_df.set_value('Avg # stationary series (FD)', 'Model ' + str(i), stationary_fd_avg)
	models_df.set_value('Avg # correlated series', 'Model ' + str(i), corr_avg)				 	  
	models_df.set_value('Avg # cointegrated series', 'Model ' + str(i), coint_avg)
	models_df.set_value('Number of trades', 'Model ' + str(i), trades)
	models_df.set_value('Profit (USD)', 'Model ' + str(i), profit)
	models_df.set_value('Gross return (%)', 'Model ' + str(i), gross_return)
	models_df.set_value('Trade success (%) [profit]', 'Model ' + str(i), pct_prof_win*100)
	models_df.set_value('Trade success (%) [disequilibrium]', 'Model ' + str(i), pct_diseq_win*100)
	models_df.set_value('Average trade duration', 'Model ' + str(i), avg_duration)
	models_df.set_value('Preformance code 1 (%)', 'Model ' + str(i), pct_pc1*100)
	models_df.set_value('Preformance code 2 (%)', 'Model ' + str(i), pct_pc2*100)
	models_df.set_value('Preformance code 3 (%)', 'Model ' + str(i), pct_pc3*100)
	models_df.set_value('Preformance code 4 (%)', 'Model ' + str(i), pct_pc4*100)
	models_df.set_value('Preformance code 5 (%)', 'Model ' + str(i), pct_pc5*100)
	models_df.set_value('Average daily change (cents)', 'Model ' + str(i), avg_daily_change*100)
	models_df.set_value('Average daily change (cents, excluding 0s)', 'Model ' + str(i), avg_daily_change_excl_0*100)
	models_df.set_value('Trade close due to convergence (%)', 'Model ' + str(i), pct_close_code_1*100)
	models_df.set_value('Trade close due to stop loss (%)', 'Model ' + str(i), pct_close_code_2*100)
	models_df.set_value('Trade close due to end of window (%)', 'Model ' + str(i), pct_close_code_3*100)
	time_split = time.clock() - loopstart
	time_split = round(time_split, 2)
	models_df.set_value('Model runtime', 'Model ' + str(i), time_split)
	return




def save_function(model_df_dict, z, models_df, setting_combinations_lst, overview_df, all_models_profit_df, HHMM, all_models_profit_fd_df, assets_df, all_models_gross_return_df, stepm_results=None, i=None):		
	save_details=overview_df.loc['Save details'][' ']
	drop_empties=overview_df.loc['Drop empties'][' ']
	safe_save=overview_df.loc['Safe save'][' ']
	current_model = i
	
	# Exporting the results to excel with the code start time (HHMM) in the file name
	name = 'Results ' + HHMM
	#writer = pd.ExcelWriter(name + '.xlsx', engine='xlsxwriter')
	path = 'C:\Users\Asus\Desktop\Output Data'
	newpath_folder = os.path.join(path,str(name))
	if not os.path.exists(newpath_folder):
		os.makedirs(newpath_folder)
	newpath_file = os.path.join(newpath_folder,str(name))
	writer = pd.ExcelWriter(newpath_file + '.xlsx', engine='xlsxwriter')
	
	workbook=writer.book
	overview_df.to_excel(writer, sheet_name='Overview')
	models_df.T.to_excel(writer, sheet_name='Models', freeze_panes=(0,models_df.T.shape[1]))
	if drop_empties == 'Y':	
		all_models_profit_df.dropna(how='all').to_excel(writer, sheet_name='AM Profits')
		all_models_profit_fd_df.dropna(how='all').to_excel(writer, sheet_name='AM Profits (FD)')
		all_models_gross_return_df=all_models_gross_return_df.dropna(how='all')
		all_models_gross_return_df.to_excel(writer, sheet_name='AM GR (%)')
	if drop_empties == 'N':
		all_models_profit_df.to_excel(writer, sheet_name='AM Profits')
		all_models_profit_fd_df.to_excel(writer, sheet_name='AM Profits (FD)')
		all_models_gross_return_df.to_excel(writer, sheet_name='AM GR (%)')
	
	if stepm_results != None:
		print
		print "Final save..."
		print
		worksheet = workbook.add_worksheet('StepM')
		model_outpreform_lst = stepm_results[0]
		worksheet.write(0, 0, 'Models which outpreform:')
		for i in range(0, len(model_outpreform_lst)):
			if len(model_outpreform_lst) != 0:
				worksheet.write(0, i+1, str(model_outpreform_lst[i]))
			if len(model_outpreform_lst) == 0:
				worksheet.write(0, 1, "None")
		worksheet.insert_image(2, 0, 'StepM.png')
		
	if save_details == 'Y':
		assets_df.to_excel(writer, sheet_name='Assets')
		if safe_save == 'N' or current_model == z:
			print
			with tqdm(total=z) as pbar:
				for j in range(0,current_model):
					if drop_empties == 'Y':
						model_df_dict[j][1][0].dropna(axis=0, how='all').to_excel(writer, sheet_name='M' + str(j) + ' ADF')
						model_df_dict[j][4][0].dropna(axis=0, how='all').to_excel(writer, sheet_name='M' + str(j) + ' CORR')
						#model_df_dict[j][2][0].dropna(axis=0, how='all').to_excel(writer, sheet_name='M' + str(j) + ' COINT_FT')
						model_df_dict[j][2][1].dropna(axis=0, how='all').to_excel(writer, sheet_name='M' + str(j) + ' COINT')
						model_df_dict[j][3][0].dropna(axis=1, how='all').to_excel(writer, sheet_name='M' + str(j) + ' TRADE PROFITS')
						#model_df_dict[j][3][1].dropna(axis=1, how='all').to_excel(writer, sheet_name='M' + str(j) + ' AT')
						trade_dict_df = pd.DataFrame.from_dict(model_df_dict[j][3][3], orient='index', dtype=None)
						trade_dict_df.columns=['Pair', 'Profit', 'Open', 'Duration', 'Close code', 'Preformance code']
						trade_dict_df.to_excel(writer, sheet_name='M' + str(j) + ' TRADE DETAILS')
						#model_df_dict[j][3][2].dropna(axis=1, how='all').to_excel(writer, sheet_name='M' + str(j) + ' DISEQ')
					if drop_empties == 'N':
						model_df_dict[j][1][0].to_excel(writer, sheet_name='M' + str(j) + ' ADF')
						model_df_dict[j][4][0].to_excel(writer, sheet_name='M' + str(j) + ' CORR')
						#model_df_dict[j][2][0].to_excel(writer, sheet_name='M' + str(j) + ' COINT_FT')	
						model_df_dict[j][2][1].to_excel(writer, sheet_name='M' + str(j) + ' COINT')
						model_df_dict[j][3][0].to_excel(writer, sheet_name='M' + str(j) + ' TRADE PROFITS')
						#model_df_dict[j][3][1].to_excel(writer, sheet_name='M' + str(j) + ' AT')
						trade_dict_df = pd.DataFrame.from_dict(model_df_dict[j][3][3], orient='index', dtype=None)
						trade_dict_df.columns=['Pair', 'Profit', 'Open', 'Duration', 'Close code', 'Preformance code']
						trade_dict_df.to_excel(writer, sheet_name='M' + str(j) + ' TRADE DETAILS')
						#model_df_dict[j][3][2].to_excel(writer, sheet_name='M' + str(j) + ' DISEQ')
					pbar.update(1)
		if safe_save == 'Y' and current_model != z:
			for j in range(0,current_model):
					if drop_empties == 'Y':
						model_df_dict[j][1][0].dropna(axis=0, how='all').to_excel(writer, sheet_name='M' + str(j) + ' ADF')
						model_df_dict[j][4][0].dropna(axis=0, how='all').to_excel(writer, sheet_name='M' + str(j) + ' CORR')
						#model_df_dict[j][2][0].dropna(axis=0, how='all').to_excel(writer, sheet_name='M' + str(j) + ' COINT_FT')	
						model_df_dict[j][2][1].dropna(axis=0, how='all').to_excel(writer, sheet_name='M' + str(j) + ' COINT')		
						model_df_dict[j][3][0].dropna(axis=1, how='all').to_excel(writer, sheet_name='M' + str(j) + ' TRADE PROFITS')
						model_df_dict[j][3][1].dropna(axis=1, how='all').to_excel(writer, sheet_name='M' + str(j) + ' AT')
						trade_dict_df = pd.DataFrame.from_dict(model_df_dict[j][3][3], orient='index', dtype=None)
						trade_dict_df.to_excel(writer, sheet_name='M' + str(j) + ' TRADE DETAILS')
						model_df_dict[j][3][2].dropna(axis=1, how='all').to_excel(writer, sheet_name='M' + str(j) + ' DISEQ')
						trade_dict_df.columns=['Pair', 'Profit', 'Open', 'Duration', 'Close code', 'Preformance code']
					if drop_empties == 'N':
						model_df_dict[j][1][0].to_excel(writer, sheet_name='M' + str(j) + ' ADF')
						model_df_dict[j][4][0].to_excel(writer, sheet_name='M' + str(j) + ' CORR')
						#model_df_dict[j][2][0].to_excel(writer, sheet_name='M' + str(j) + ' COINT_FT')	
						model_df_dict[j][2][1].to_excel(writer, sheet_name='M' + str(j) + ' COINT')
						model_df_dict[j][3][0].to_excel(writer, sheet_name='M' + str(j) + ' TRADE PROFITS')
						model_df_dict[j][3][1].to_excel(writer, sheet_name='M' + str(j) + ' AT')
						trade_dict_df = pd.DataFrame.from_dict(model_df_dict[j][3][3], orient='index', dtype=None)
						trade_dict_df.columns=['Pair', 'Profit', 'Open', 'Duration', 'Close code', 'Preformance code']
						trade_dict_df.to_excel(writer, sheet_name='M' + str(j) + ' TRADE DETAILS')
						#model_df_dict[j][3][2].to_excel(writer, sheet_name='M' + str(j) + ' DISEQ')
	
	writer.save()
	overview_df.to_csv(newpath_file+' overview.csv')
	models_df.T.to_csv(newpath_file+' models.csv')
	return





def multiple_testing(all_models_profit_fd_df, HHMM):
	all_models_LOSSES_fd_df=-all_models_profit_fd_df
	benchmark_df = pd.DataFrame(0, index=all_models_LOSSES_fd_df.index, columns=['benchmark'])
	avg_model_losses = pd.DataFrame(all_models_LOSSES_fd_df.mean(0), columns=['Average loss'])
	
	# Null hypothesis: The benchmark is not inferior to any of the alternatives
	
	# Tried using SPA p-value to coax StepM into churning out more than one p-value, but broken
	"""spa = SPA(benchmark_df, all_models_LOSSES_fd_df)
	spa.compute()"""

	
	stepm = StepM(benchmark_df, all_models_LOSSES_fd_df, size=0.10)
	stepm_pvals_df = stepm.compute()
	
	spa = SPA(benchmark_df, all_models_LOSSES_fd_df)
	spa.compute()
	
	
	
	# Very inefficient code that forces StepM function to loop with different target p-values until the p-values for all of the models are calculated
	# Never managed to make this work
	"""while stepm_pvals_df.shape[0] < all_models_LOSSES_fd_df.shape[1]:
		new_pval_ratchet = stepm_pvals_df.loc[:,'Consistent'].max()
		stepm = StepM(benchmark_df, all_models_LOSSES_fd_df, new_pval_ratchet)
		stepm_pvals_df = stepm.compute()
		print stepm_pvals_df"""
	

	model_pvals=avg_model_losses.sort_values(by=['Average loss'])
	model_pvals.set_value(model_pvals.index[0],'StepM P-value', stepm_pvals_df.iloc[0,1])
	model_pvals.set_value(model_pvals.index[0],'SPA P-value', spa.pvalues[1])
	model_pvals = model_pvals.dropna(how='any')
	print 
	print model_pvals
	print
	
	stepm = StepM(benchmark_df, all_models_LOSSES_fd_df, 0.10)
	stepm_pvals_df = stepm.compute()
	model_outpreform_lst=[model.split('.')[1] for model in stepm.superior_models]
	if len(model_outpreform_lst) > 0:
		survive = model_outpreform_lst
	if len(model_outpreform_lst) == 0:
		survive = "None"
	print
	print('Indices of models which survive multiple testing correction: %s') % (survive)
	better_models = pd.concat([all_models_LOSSES_fd_df.mean(0),all_models_LOSSES_fd_df.mean(0)],1)
	better_models.columns = ['Same or worse','Better']
	better = better_models.index.isin(stepm.superior_models)
	worse = np.logical_not(better)
	better_models.loc[better,'Same or worse'] = np.nan
	better_models.loc[worse,'Better'] = np.nan
	fig = better_models.plot(style=['o','s'], rot=270)
	
	plt.savefig('StepM.png', bbox_inches='tight')
	plt.title("\nAverage Daily Model Losses\n")
	plt.show()
	
	return(model_outpreform_lst, fig, model_pvals)
	
	
	
	
	
def shim(all_models_profit_fd_df):
	import random
	random_returns = []
	for i in range(0,len(all_models_profit_fd_df)):
		random_returns.append(random.gauss(-0.1,0.5))
	shim_df = pd.DataFrame(random_returns, index=all_models_profit_fd_df.index)
	return shim_df





quickload()