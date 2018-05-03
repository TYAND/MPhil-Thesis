# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 09:25:28 2018

@author: Tim
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 17:16:48 2018

@author: Tim
"""

import sys
import itertools
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import datetime
import numpy as np
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from collections import Counter
plt.rcParams["figure.figsize"] = (24,35)
#plt.rcParams["figure.figsize"] = (12,8)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set()
sns.set_style('ticks')
sns.despine()

input_filename = 'Results 1951'
data_df = pd.read_csv("C:\Users\Asus\Desktop\Output Data\Results (Final)\Symmetric legs\Beta vs Neutral.csv")
#overview_df = pd.read_csv("C:\Users\Asus\Desktop\Output Data\Results 1151\Results 1151 overview.csv")


#assets = pd.read_csv('C:\Users\Asus\Desktop\Input Data\USD_Currency_Pairs $ (23, 10k centered, droprows & fwd fill).csv')
#assets = assets.dropna(how='all', axis=0).dropna(how='all', axis=1)
#assets_df = pd.DataFrame(data = assets)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#data_df=data_df.shift(periods=1, axis=0)
data_df.index=np.arange(1,49,1)
print data_df
#print data_df

# Key:

#model = data_df.iloc[:,0]
#adf_pvalue_threshold = data_df.iloc[:,1]	
#correlation_pvalue_threshold = data_df.iloc[:,2]
#cointegration_pvalue_threshold = data_df.iloc[:,3]
#window_size = data_df.iloc[:,4]

#open_trigger_sensitivity = data_df.iloc[:,5]
#close_trigger_sensitivity = data_df.iloc[:,6]
#stop_loss_trigger_sensitivity = data_df.iloc[:,7]

import ast

#OTS_lst = ast.literal_eval(overview_df.iloc[13].values[1]) # Open trigger sensitivity
#CTS_lst = ast.literal_eval(overview_df.iloc[14].values[1]) # Close trigger sensitivity
#STS_lst = ast.literal_eval(overview_df.iloc[15].values[1]) # Stop trigger sensitivity

#triggers = pd.DataFrame(columns = ['Open', 'Close', 'Stop'])

#for j in range (0,len(OTS_lst)):
#	triggers.set_value(j, 'Open', OTS_lst[j])
#	triggers.set_value(j, 'Close', CTS_lst[j])
#	triggers.set_value(j, 'Stop', STS_lst[j])

#stop_loss_type = data_df.iloc[:,8]
#asymmetric_legs	= data_df.iloc[:,9]
#drop_adf_failures = data_df.iloc[:,10]
#correlation_prescreen = data_df.iloc[:,11]
#cointegration_test = data_df.iloc[:,12]
#include_constant = data_df.iloc[:,13]
#avg_num_stationary_series_levels = data_df.iloc[:,14]
#avg_num_stationary_series_fd = data_df.iloc[:,15]
#avg_num_correlated_series = data_df.iloc[:,16]
#avg_num_cointegrated_series = data_df.iloc[:,17]
#number_of_trades = data_df.iloc[:,18]
#profit_usd = data_df.iloc[:,19] 
#trade_success_pct_profit = data_df.iloc[:,20]
#trade_success_pct_disequilibrium = data_df.iloc[:,21] 
#average_trade_duration = data_df.iloc[:,22]
#preformance_code_1_pct = data_df.iloc[:,23]
#preformance_code_2_pct = data_df.iloc[:,24]
#preformance_code_3_pct = data_df.iloc[:,25] 
#preformance_code_4_pct = data_df.iloc[:,26]
#average_daily_change_cents = data_df.iloc[:,27]
#average_daily_change_cents_excluding_0s = data_df.iloc[:,28]
#trade_close_due_to_convergence_pct = data_df.iloc[:,29]
#trade_close_due_to_stop_loss_pct = data_df.iloc[:,30]
#trade_close_due_to_end_of_window_pct = data_df.iloc[:,31]
#model_runtime = data_df.iloc[:,32]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

pp = PdfPages(input_filename+'.pdf')


#num_series_df = pd.DataFrame(data=None, columns=['Day', 'Month', 'Year', 'Date', 'Datetime', 'Series Count'])
#num_series_df.loc[:,'Day'] = assets_df.loc[:,'Day']
#num_series_df.loc[:,'Month'] = assets_df.loc[:,'Month'].fillna(method='ffill')
#num_series_df.loc[:,'Year'] = assets_df.loc[:,'Year'].fillna(method='ffill')
#num_series_df.loc[:,'Date'] = assets_df.loc[:,'DATE']
#num_series_df.loc[:,'Series Count'] = assets_df.loc[:,'BZUS':].count(axis=1)

fig, axes = plt.subplots(5,1)
plt.subplots_adjust(wspace = 0.2, hspace = 0.20)
from matplotlib import rcParams
rcParams['axes.titlepad'] = 10
#plt.tight_layout()
for i in [7]:
	ax_y=0
	for j in [16, 19, 17, 18, 21]:
				
			# 'i' is the setting, 'j' is the preformance metric
			x = data_df.iloc[:,i].values
			y = data_df.iloc[:,j].values
			xy = np.column_stack((x,y))
			print xy
			xy = pd.DataFrame(data=xy, columns=[str(data_df.columns.values[i]),str(data_df.columns.values[j])], index=data_df.index)
			xy.sort_values(str(data_df.columns.values[i]), inplace=True)
			length = len(Counter(x).values())
			
	
			if length == 1:
				pass
				
			elif length > 1:
				stats_df=pd.DataFrame(data=None)
				columns_lst=[]
				
				split_df = np.split(xy, length, axis=0)
				stats_df = pd.DataFrame(data=None, index=["Mean", "SD"], columns=columns_lst)
				l=(["P-value Threshold = 0.01", "P-value Threshold = 0.05", "P-value Threshold = 0.10"],["Window Size = 250 Days", "Window Size = 500 Days"])
				label_lst=["Dollar Neutral Legs", "Cointegrating Coefficient Weighted Legs"]
				for z in range(0,length):
					colours=(sns.xkcd_rgb["medium blue"], sns.xkcd_rgb["black"], sns.xkcd_rgb["red orange"], sns.xkcd_rgb["grey"])
					axes[ax_y].scatter(split_df[z].index, split_df[z].iloc[:,1], c=colours[z], label=label_lst[z], s=100)
					axes[ax_y].set_title("("+str(ax_y)+")", size=18)
					for m in range(3,49,6):
						axes[ax_y].axvline(x=m, linestyle='-', alpha=0.05, c=sns.xkcd_rgb["black"])
					axes[ax_y].set_ylabel(str(data_df.columns.values[j]), size=18)
					axes[ax_y].get_yaxis().set_label_coords(-0.05,0.5)
					if ax_y==4:
						axes[ax_y].set_xlabel("Strategy", size=18)
						axes[ax_y].get_xaxis().set_label_coords(0.5,-0.15)
					columns_lst.append(str(split_df[z].iloc[0,0]))
					stats_df.loc['Mean', str(split_df[z].iloc[0,0])] = np.mean(split_df[z].iloc[:,1])
					stats_df.loc['SD', str(split_df[z].iloc[0,0])] = np.std(split_df[z].iloc[:,1])
					major_ticks = np.arange(1, 49, 2)
					axes[ax_y].tick_params(axis = 'both', which = 'major', labelsize = 14)
					axes[ax_y].set_xticks(major_ticks)
				
				legend_lst = []
				for z in range(0,length):
					legend_lst.append(str(split_df[0].columns.values[0])+" = "+str(split_df[z].iloc[0,0]))
				
				for z in range(0,length):
					axes[ax_y].axhline(y=stats_df.iloc[0,z], xmin=0, xmax=49, c=colours[z], linestyle='--', alpha=0.75)
				
				#axes[ax_y,1].set_ticks_position('both')
				#legend_lst = []
				#for z in range(0,length):
				#	legend_lst.append(str(split_df[0].columns.values[0])+" = "+str(split_df[z].iloc[0,0]))
				if ax_y == 0:
					axes[0].legend(bbox_to_anchor=(0., 1.1, 1., .102), loc=9,
           ncol=2, mode="expand", borderaxespad=0, prop={'size': 16})
				#axes[ax_y].ylabel=(str(data_df.columns.values[j]))
	
				#plt.xlabel('Strategy')
				#title = "\n%s v %s\n\n%s\n" % (str(data_df.columns.values[i]), str(data_df.columns.values[j]), stats_df.to_string())
				#plt.title(title)
				#figure = plt.gcf()
				#pp.savefig(figure)
			ax_y+=1
	#plt.xlim([0.5,24.5])
plt.show()