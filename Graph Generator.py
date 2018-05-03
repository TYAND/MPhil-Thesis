# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 19:40:09 2018

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
data_df = pd.read_csv("C:\Users\Asus\Desktop\Output Data\Results 1151\Results 1151 models.csv")
overview_df = pd.read_csv("C:\Users\Asus\Desktop\Output Data\Results 1151\Results 1151 overview.csv")

assets = pd.read_csv('C:\Users\Asus\Desktop\Input Data\USD_Currency_Pairs $ (23, 10k centered, droprows & fwd fill).csv')
assets = assets.dropna(how='all', axis=0).dropna(how='all', axis=1)
assets_df = pd.DataFrame(data = assets)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#data_df=data_df.shift(periods=1, axis=0)
data_df.index=np.arange(1,25,1)

#print data_df

# Key:

model = data_df.iloc[:,0]
adf_pvalue_threshold = data_df.iloc[:,1]	
correlation_pvalue_threshold = data_df.iloc[:,2]
cointegration_pvalue_threshold = data_df.iloc[:,3]
window_size = data_df.iloc[:,4]

open_trigger_sensitivity = data_df.iloc[:,5]
close_trigger_sensitivity = data_df.iloc[:,6]
stop_loss_trigger_sensitivity = data_df.iloc[:,7]

import ast

OTS_lst = ast.literal_eval(overview_df.iloc[13].values[1]) # Open trigger sensitivity
CTS_lst = ast.literal_eval(overview_df.iloc[14].values[1]) # Close trigger sensitivity
STS_lst = ast.literal_eval(overview_df.iloc[15].values[1]) # Stop trigger sensitivity

triggers = pd.DataFrame(columns = ['Open', 'Close', 'Stop'])

for j in range (0,len(OTS_lst)):
	triggers.set_value(j, 'Open', OTS_lst[j])
	triggers.set_value(j, 'Close', CTS_lst[j])
	triggers.set_value(j, 'Stop', STS_lst[j])

stop_loss_type = data_df.iloc[:,8]
asymmetric_legs	= data_df.iloc[:,9]
drop_adf_failures = data_df.iloc[:,10]
correlation_prescreen = data_df.iloc[:,11]
cointegration_test = data_df.iloc[:,12]
include_constant = data_df.iloc[:,13]
avg_num_stationary_series_levels = data_df.iloc[:,14]
avg_num_stationary_series_fd = data_df.iloc[:,15]
avg_num_correlated_series = data_df.iloc[:,16]
avg_num_cointegrated_series = data_df.iloc[:,17]
number_of_trades = data_df.iloc[:,18]
profit_usd = data_df.iloc[:,19] 
trade_success_pct_profit = data_df.iloc[:,20]
trade_success_pct_disequilibrium = data_df.iloc[:,21] 
average_trade_duration = data_df.iloc[:,22]
preformance_code_1_pct = data_df.iloc[:,23]
preformance_code_2_pct = data_df.iloc[:,24]
preformance_code_3_pct = data_df.iloc[:,25] 
preformance_code_4_pct = data_df.iloc[:,26]
average_daily_change_cents = data_df.iloc[:,27]
average_daily_change_cents_excluding_0s = data_df.iloc[:,28]
trade_close_due_to_convergence_pct = data_df.iloc[:,29]
trade_close_due_to_stop_loss_pct = data_df.iloc[:,30]
trade_close_due_to_end_of_window_pct = data_df.iloc[:,31]
model_runtime = data_df.iloc[:,32]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

pp = PdfPages(input_filename+'.pdf')


num_series_df = pd.DataFrame(data=None, columns=['Day', 'Month', 'Year', 'Date', 'Datetime', 'Series Count'])
num_series_df.loc[:,'Day'] = assets_df.loc[:,'Day']
num_series_df.loc[:,'Month'] = assets_df.loc[:,'Month'].fillna(method='ffill')
num_series_df.loc[:,'Year'] = assets_df.loc[:,'Year'].fillna(method='ffill')
num_series_df.loc[:,'Date'] = assets_df.loc[:,'DATE']
num_series_df.loc[:,'Series Count'] = assets_df.loc[:,'BZUS':].count(axis=1)
"""
for i in range (0,assets_df.shape[0]):
	year = int(num_series_df.loc[i,'Year'])
	month = int(num_series_df.loc[i,'Month'])
	day = int(num_series_df.loc[i,'Day'])
	num_series_df.loc[i,'Datetime'] = datetime.datetime(year, month, day)

fig, ax = plt.subplots()
ax.plot(num_series_df.loc[:,'Datetime'].values, num_series_df.loc[:,'Series Count'].values)
ax.xaxis_date()
myFmt = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(myFmt)
fig.autofmt_xdate()
plt.xlabel('Year')
plt.ylabel('Number of Series')

for j in range(6,assets_df.shape[1]):
	name = assets_df.columns.values[j]
	print name
	fig, ax = plt.subplots()
	ax.set_xlim(left=num_series_df.loc[0,'Datetime'], emit=True, auto=False)
	ax.plot(num_series_df.loc[:,'Datetime'].values, assets_df.iloc[:,j], linewidth=5)
	#ax.set_xlim(left=num_series_df.loc[0,'Datetime'], emit=True, auto=False)
	ax.set_axis_off()
	plt.savefig(name+'.png', bbox_inches='tight')
	plt.show()

models_lst=[]
for i in range(0, len(data_df.iloc[:,0])):
	models_lst.append(i)
models=pd.DataFrame(data=models_lst)


#0. Basic scatterplots
for i in range(18,36):
	x = models.values
	y = data_df.iloc[:,i].values
	plt.scatter(x,y)
	plt.xlabel('Model')
	plt.ylabel(str(data_df.columns.values[i]))
	title = "\n%s\n" % (str(data_df.columns.values[i]))
	plt.title(title)
	figure = plt.gcf()
	pp.savefig(figure)
	plt.show()


# 1. Basic histograms
for i in range(18,23):
	x = data_df.iloc[:,i].values
	plt.hist(x, density=1)
	N, bins, patches = plt.hist(x, density=1)
	
	# Colouring bins by their frequency
	fracs = N / N.max()
	norm = colors.Normalize(fracs.min(), fracs.max())
	for thisfrac, thispatch in zip(fracs, patches):
	    color = plt.cm.viridis(norm(thisfrac))
	    thispatch.set_facecolor(color)
	
	mu = np.mean(x)
	sigma = np.std(x)
	z = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
	plt.plot(bins, z, '--', color = (0.8, 0.1, 0.1))
	
	plt.xlabel(str(data_df.columns.values[i]))
	plt.ylabel("Density")	
	title = "\nDistribution: %s\n\nMu: %s        Sigma: %s\n" % (str(data_df.columns.values[i]), round(mu,3), round(sigma,3))
	plt.title(title)
	figure = plt.gcf()
	pp.savefig(figure)
	plt.show()
"""
"""
# 2. Preformance metric vs preformance metric scatterplots
lst = []
for i in range(18,35):
	lst.append(i)
combinations = list(itertools.combinations(lst, 2))

for z in combinations:	
	i=z[0]
	j=z[1]
	x = data_df.iloc[:,i].values
	y = data_df.iloc[:,j].values
	plt.scatter(x, y)
	# Computing line of best fit
	slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
	z = np.polyfit(x, y, 1)
	p = np.poly1d(z)
	plt.plot(x,p(x),"r--")
	equation = "y=%.3fx+%.3f" % (z[0],z[1])
	plt.xlabel(str(data_df.columns.values[i]))
	plt.ylabel(str(data_df.columns.values[j]))
	title = "\n%s v %s\n\nEquation: %s        R-squared: %s        Slope-coefficient p-value: %s\n" % (str(data_df.columns.values[i]), str(data_df.columns.values[j]), equation, round(r_value**2, 3), round(p_value,3))
	plt.title(title)
	figure = plt.gcf()
	pp.savefig(figure)
	plt.show()"""

"""# 3. Overlaid histograms to show how outcomes are impacted by settings 
for i in range(3,14):
	for j in range(18,23):
		# 'i' is the setting, 'j' is the preformance metric
		x = data_df.iloc[:,i].values
		y = data_df.iloc[:,j].values
		xy = np.column_stack((x,y))
		xy = pd.DataFrame(data=xy, columns=[str(data_df.columns.values[i]),str(data_df.columns.values[j])])
		xy.sort_values(str(data_df.columns.values[i]), inplace=True)
		length = len(Counter(x).values())
		
		if length == 1:
			pass
		
		elif length == 2:
			one = xy.iloc[0:(xy.shape[0]/2),1]
			two = xy.iloc[(xy.shape[0]/2):,1] 
			both = np.column_stack((one,two))
			
			n, bins, patches = plt.hist(both, histtype ='step', color = [(0.1, 0.1, 0.8),(0.8, 0.1, 0.1)], density=1)
			stats_df = pd.DataFrame(data=None, index=["Mean", "SD"], columns=[str(xy.iloc[0,0]),str(xy.iloc[-1,0])])
			
			mean = np.mean(one)
			SD = np.std(one)
			z = ((1 / (np.sqrt(2 * np.pi) * SD)) * np.exp(-0.5 * (1 / SD * (bins - mean))**2))
			plt.plot(bins, z, '--', color = (0.1, 0.1, 0.8))
			stats_df.loc['Mean',str(xy.iloc[0,0])]=round(mean, 3)
			stats_df.loc['SD',str(xy.iloc[0,0])]=round(SD, 3)
			
			mean = np.mean(two)
			SD = np.std(two)
			w = ((1 / (np.sqrt(2 * np.pi) * SD)) * np.exp(-0.5 * (1 / SD * (bins - mean))**2))
			plt.plot(bins, w, '--', color = (0.8, 0.1, 0.1))
			stats_df.loc['Mean',str(xy.iloc[-1,0])]=round(mean, 3)
			stats_df.loc['SD',str(xy.iloc[-1,0])]=round(SD, 3)
			
			plt.legend([str(data_df.columns.values[i])+" = "+str(data_df.iloc[0,i]), str(data_df.columns.values[i])+" =  "+str(data_df.iloc[-1,i])])
			plt.xlabel(str(data_df.columns.values[j]))
			plt.ylabel('Density')
			
			title = "\n%s v %s\n\n%s\n" % (str(data_df.columns.values[i]), str(data_df.columns.values[j]), stats_df.to_string())
			plt.title(title)
			figure = plt.gcf()
			pp.savefig(figure)
			plt.show()
			
		elif length == 3:
			one = xy.iloc[0:(xy.shape[0]/3),1]
			two = xy.iloc[(xy.shape[0]/3):(2*xy.shape[0]/3),1]
			three = xy.iloc[(2*xy.shape[0]/3):,1]
			threeoth = np.column_stack((one,two,three))
			
			n, bins, patches = plt.hist(threeoth, histtype ='step', color = [(0.1, 0.1, 0.8),(0.8, 0.1, 0.1), (0.1, 0.8, 0.1)], density=1)
			stats_df = pd.DataFrame(data=None, index=["Mean", "SD"], columns=[str(xy.iloc[0,0]), str(xy.iloc[(len(data_df)/2),0]), str(xy.iloc[-1,0])])
			
			mean = np.mean(one)
			SD = np.std(one)
			z = ((1 / (np.sqrt(2 * np.pi) * SD)) * np.exp(-0.5 * (1 / SD * (bins - mean))**2))
			plt.plot(bins, z, '--', color = (0.1, 0.1, 0.8))
			stats_df.loc['Mean',str(xy.iloc[0,0])]=round(mean, 3)
			stats_df.loc['SD',str(xy.iloc[0,0])]=round(SD, 3)
			
			mean = np.mean(two)
			SD = np.std(two)
			w = ((1 / (np.sqrt(2 * np.pi) * SD)) * np.exp(-0.5 * (1 / SD * (bins - mean))**2))
			plt.plot(bins, w, '--', color = (0.8, 0.1, 0.1))
			stats_df.loc['Mean',str(xy.iloc[len(data_df)/2,0])]=round(mean, 3)
			stats_df.loc['SD',str(xy.iloc[len(data_df)/2,0])]=round(SD, 3)
			
			mean = np.mean(three)
			SD = np.std(three)
			v = ((1 / (np.sqrt(2 * np.pi) * SD)) * np.exp(-0.5 * (1 / SD * (bins - mean))**2))
			plt.plot(bins, v, '--', color = (0.1, 0.8, 0.1))
			stats_df.loc['Mean',str(xy.iloc[-1,0])]=round(mean,3)
			stats_df.loc['SD',str(xy.iloc[-1,0])]=round(SD,3)
			
			plt.legend([str(xy.columns.values[0])+" = "+str(xy.iloc[0,0]), str(xy.columns.values[0])+" = "+str(xy.iloc[len(xy)/2,0]), str(xy.columns.values[0])+" =  "+str(xy.iloc[-1,0])])
			plt.xlabel(str(data_df.columns.values[j]))
			plt.ylabel('Density')
		
			title = "\n%s v %s\n\n%s\n" % (str(data_df.columns.values[i]), str(data_df.columns.values[j]), stats_df.to_string())
			plt.title(title)
			figure = plt.gcf()
			pp.savefig(figure)
			plt.show()"""


# 4A. Overlaid scatterplots to show how outcomes are impacted by settings (excluding triggers, which need special treatment)
fig, axes = plt.subplots(5,3)
plt.subplots_adjust(wspace = 0.2, hspace = 0.20)
from matplotlib import rcParams
rcParams['axes.titlepad'] = 10
#plt.tight_layout()
ax_x = 0
for i in [3, 4]:
	ax_y=0
	for j in [18,19,20,21,23]:
		if not i in [5,6,7]:
				
			# 'i' is the setting, 'j' is the preformance metric
			x = data_df.iloc[:,i].values
			y = data_df.iloc[:,j].values
			xy = np.column_stack((x,y))
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
				ll=i-3
				for z in range(0,length):
					colours=(sns.xkcd_rgb["medium blue"], sns.xkcd_rgb["black"], sns.xkcd_rgb["red orange"], sns.xkcd_rgb["grey"])
					axes[ax_y,ax_x].scatter(split_df[z].index, split_df[z].iloc[:,1], c=colours[z], label=l[ll][z], s=100)
					axes[ax_y,ax_x].set_title("("+str(ax_y)+", "+str(ax_x)+")", size=18)
					for m in range(3,25,6):
						axes[ax_y,ax_x].axvline(x=m, linestyle='-', alpha=0.05, c=sns.xkcd_rgb["black"])
					if ax_x==0:
						axes[ax_y,ax_x].set_ylabel(str(data_df.columns.values[j]), size=18)
						axes[ax_y,ax_x].get_yaxis().set_label_coords(-0.2,0.5)
					if ax_y==4:
						axes[ax_y,ax_x].set_xlabel("Strategy", size=18)
						axes[ax_y,ax_x].get_xaxis().set_label_coords(0.5,-0.2)
					columns_lst.append(str(split_df[z].iloc[0,0]))
					stats_df.loc['Mean', str(split_df[z].iloc[0,0])] = np.mean(split_df[z].iloc[:,1])
					stats_df.loc['SD', str(split_df[z].iloc[0,0])] = np.std(split_df[z].iloc[:,1])
					major_ticks = np.arange(1, 25, 2)
					axes[ax_y,ax_x].tick_params(axis = 'both', which = 'major', labelsize = 14)
					axes[ax_y,ax_x].set_xticks(major_ticks)
				
				legend_lst = []
				for z in range(0,length):
					legend_lst.append(str(split_df[0].columns.values[0])+" = "+str(split_df[z].iloc[0,0]))
				
				for z in range(0,length):
					axes[ax_y,ax_x].axhline(y=stats_df.iloc[0,z], xmin=0, xmax=24, c=colours[z], linestyle='--', alpha=0.75)
				
				#axes[ax_y,1].set_ticks_position('both')
				#legend_lst = []
				#for z in range(0,length):
				#	legend_lst.append(str(split_df[0].columns.values[0])+" = "+str(split_df[z].iloc[0,0]))
				if ax_y == 0:
					axes[0,ax_x].legend(bbox_to_anchor=(0., 1.3, 1., .102), loc=9,
           ncol=1, mode="expand", borderaxespad=0, prop={'size': 16})
				#axes[ax_y].ylabel=(str(data_df.columns.values[j]))
	
				#plt.xlabel('Strategy')
				#title = "\n%s v %s\n\n%s\n" % (str(data_df.columns.values[i]), str(data_df.columns.values[j]), stats_df.to_string())
				#plt.title(title)
				#figure = plt.gcf()
				#pp.savefig(figure)
			ax_y+=1
	plt.xlim([0.5,24.5])
	ax_x+=1

# 4B. Overlaid scatterplots to show how outcomes are impacted by settings for triggers
trigger_sort_df = data_df.sort_values(by=['Open trigger sensitivity', 'Close trigger sensitivity', 'Stop loss trigger sensitivity'])
trigger_sort_df['Stop loss trigger sensitivity']=trigger_sort_df['Stop loss trigger sensitivity'].fillna('N')
trigger_sort_df_split = np.split(trigger_sort_df, 4, axis=0)
stats_df = pd.DataFrame(data=None, index=["Mean", "SD"])					
legend_lst=[]

#sys.exit()
#print trigger_sort_df_split[i].index

#fig, axes = plt.subplots(5,1, sharex=True)
ax_y=0

"""
for j in range(18,24):
	colours=(sns.xkcd_rgb["medium blue"], sns.xkcd_rgb["red orange"], sns.xkcd_rgb["black"], sns.xkcd_rgb["greenish"])
	labels=["Triggers = 2,0,3","Triggers = 2,0,N"," Triggers = 2,1,3","Triggers = 2,1,N"]
	for i in range(0, len(trigger_sort_df_split)):
		stats_df.loc['Mean', "["+str(trigger_sort_df_split[i].iloc[0,5])+","+str(trigger_sort_df_split[i].iloc[0,6])+","+str(trigger_sort_df_split[i].iloc[0,7])+"]"] = np.mean(trigger_sort_df_split[i].iloc[:,j])
		stats_df.loc['SD', "["+str(trigger_sort_df_split[i].iloc[0,5])+","+str(trigger_sort_df_split[i].iloc[0,6])+","+str(trigger_sort_df_split[i].iloc[0,7])+"]"] = np.std(trigger_sort_df_split[i].iloc[:,j])
		mkr_lst=['x','o','x','o','x','o']
		for g in range(0,trigger_sort_df_split[i].iloc[:,j].shape[0]-1):
			#print trigger_sort_df_split[i].iloc[:,j].shape[0]-1
			#print g
			axes[ax_y].scatter(trigger_sort_df_split[i].index[g], trigger_sort_df_split[i].iloc[g,j], c=colours[i], label=labels[i], marker=mkr_lst[g])
		axes[ax_y].set_ylabel(str(data_df.columns.values[j]))
		legend_lst.append(str(trigger_sort_df_split[i].iloc[0,5])+", "+str(trigger_sort_df_split[i].iloc[0,6])+", "+str(trigger_sort_df_split[i].iloc[0,7]))
	"""
	
for j in [18,19,20,21,23]:
	colours=(sns.xkcd_rgb["medium blue"], sns.xkcd_rgb["red orange"], sns.xkcd_rgb["black"], sns.xkcd_rgb["greenish"])
	labels=["Triggers = 2,0,3","Triggers = 2,0,N"," Triggers = 2,1,3","Triggers = 2,1,N"]
	mkr_lst=['o','X','o','X']
	for i in range(0, len(trigger_sort_df_split)):
		stats_df.loc['Mean', "["+str(trigger_sort_df_split[i].iloc[0,5])+","+str(trigger_sort_df_split[i].iloc[0,6])+","+str(trigger_sort_df_split[i].iloc[0,7])+"]"] = np.mean(trigger_sort_df_split[i].iloc[:,j])
		stats_df.loc['SD', "["+str(trigger_sort_df_split[i].iloc[0,5])+","+str(trigger_sort_df_split[i].iloc[0,6])+","+str(trigger_sort_df_split[i].iloc[0,7])+"]"] = np.std(trigger_sort_df_split[i].iloc[:,j])
		axes[ax_y,2].scatter(trigger_sort_df_split[i].index, trigger_sort_df_split[i].iloc[:,j], c=colours[i], label=labels[i], marker=mkr_lst[i], s=100)
		#axes[ax_y,2].set_ylabel(str(data_df.columns.values[j]))
		legend_lst.append(str(trigger_sort_df_split[i].iloc[0,5])+", "+str(trigger_sort_df_split[i].iloc[0,6])+", "+str(trigger_sort_df_split[i].iloc[0,7]))
		axes[ax_y,ax_x].tick_params(axis = 'both', which = 'major', labelsize = 14)
		axes[ax_y,ax_x].set_title("("+str(ax_y)+", "+str(ax_x)+")", size=18)
		major_ticks = np.arange(1, 25, 2)
		for m in range(3,25,6):
			axes[ax_y,2].axvline(x=m, linestyle='-', alpha=0.05, c=sns.xkcd_rgb["black"])
		axes[ax_y,2].set_xticks(major_ticks)
	for t in range(0,stats_df.shape[0]):
		for p in range(0,stats_df.shape[1]):
			stats_df.iloc[t,p]=round(stats_df.iloc[t,p],2)
	for z in range(0,4):
					axes[ax_y,2].axhline(y=stats_df.iloc[0,z], xmin=0, xmax=24, c=colours[z], linestyle='--', alpha=0.75)
	#title = "\nTriggers v %s\n\n%s\n" % (str(data_df.columns.values[j]), stats_df.to_string())
	#plt.title(title)
	axes[0,2].legend(bbox_to_anchor=(0., 1.3, 1., .102), loc=9,
           ncol=2, mode="expand", borderaxespad=0, prop={'size': 16})
	#plt.legend(legend_lst)
	#axes[ax_y,2].ylabel(str(data_df.columns.values[j]))
	#axes[ax_y,2].yaxis.tick_right()
	#axes[ax_y,2].tick_params(axis='y', which='both', labelleft='off', labelright='on')
	plt.xlabel('Strategy', size=18)
	axes[ax_y,ax_x].get_xaxis().set_label_coords(0.5,-0.2)
	plt.xlim([0.5,24.5])
	#figure = plt.gcf()
	#pp.savefig(figure)
	ax_y+=1
	
plt.show()
	
		
pp.close()