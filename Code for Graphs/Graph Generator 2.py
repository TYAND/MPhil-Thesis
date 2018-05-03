# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:12:55 2018

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
import matplotlib.dates as dates
import numpy as np
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from collections import Counter
plt.rcParams["figure.figsize"] = (22,15)
#plt.rcParams["figure.figsize"] = (6,6)
#plt.rcParams["figure.figsize"] = (18,30) # Seems optimal for full-page big boi graphs
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set()
sns.set_style('ticks')
sns.despine()


am_profits = pd.read_csv("C:\Users\Asus\Desktop\Output Data\Results 1151\AM Profits.csv", index_col=1, parse_dates=True, dayfirst=True)
am_profits_fd=am_profits.diff()

am_gr = pd.read_csv("C:\Users\Asus\Desktop\Output Data\Results 1151\AM GR.csv", index_col=1, parse_dates=True, dayfirst=True)
am_gr_fd=am_gr.diff()

am_trades = pd.read_csv("C:\Users\Asus\Desktop\Output Data\Results 1151\AM Trades.csv")




#
# Graph 1
# Subplots showing the levels and first differences of profit and return on capital employed over time for the best and worst models, and the average of all 24 
#

fig, axes = plt.subplots(2,2, sharex=True)


maxcol=am_gr.iloc[-1,1:].idxmax()
mincol=am_gr.iloc[-1,1:].idxmin()
axes[0,0].plot(am_gr.index, am_gr[mincol], color=sns.xkcd_rgb["red orange"], label="Worst Strategy")
axes[0,0].plot(am_gr.index, am_gr.iloc[:,1:].mean(1), color=sns.xkcd_rgb["black"], label="Average")
axes[0,0].plot(am_gr.index, am_gr[maxcol], color=sns.xkcd_rgb["medium blue"], label="Best Strategy")
axes[0,0].legend(fontsize=18)
axes[0,0].set_ylabel("Levels", size=24)
axes[0,0].set_title("Return on Capital Employed (%)", size=24)

maxcol=am_gr.iloc[-1,1:].idxmax()
mincol=am_gr.iloc[-1,1:].idxmin()
axes[1,0].plot(am_gr_fd.index, am_gr_fd[mincol], color=sns.xkcd_rgb["red orange"])
axes[1,0].plot(am_gr_fd.index, am_gr_fd.iloc[:,1:].mean(1), color=sns.xkcd_rgb["black"])
axes[1,0].plot(am_gr_fd.index, am_gr_fd[maxcol], color=sns.xkcd_rgb["medium blue"])
axes[1,0].set_ylabel("First Differences", size=24)

maxcol=am_profits.iloc[-1,1:].idxmax()
mincol=am_profits.iloc[-1,1:].idxmin()
axes[0,1].plot(am_profits.index, am_profits[mincol], color=sns.xkcd_rgb["red orange"])
axes[0,1].plot(am_profits.index, am_profits.iloc[:,1:].mean(1), color=sns.xkcd_rgb["black"])
axes[0,1].plot(am_profits.index, am_profits[maxcol], color=sns.xkcd_rgb["medium blue"])
axes[0,1].set_title("Profit (USD)", size=24)

maxcol=am_gr.iloc[-1,1:].idxmax()
mincol=am_gr.iloc[-1,1:].idxmin()
axes[1,1].plot(am_profits_fd.index, am_profits_fd[mincol], color=sns.xkcd_rgb["red orange"])
axes[1,1].plot(am_profits_fd.index, am_profits_fd.iloc[:,1:].mean(1), color=sns.xkcd_rgb["black"])
axes[1,1].plot(am_profits_fd.index, am_profits_fd[maxcol], color=sns.xkcd_rgb["medium blue"])

axes[0,0].tick_params(axis='both', which='major', labelsize=18)
axes[1,0].tick_params(axis='both', which='major', labelsize=18)
axes[0,1].tick_params(axis='both', which='major', labelsize=18)
axes[1,1].tick_params(axis='both', which='major', labelsize=18)

plt.xlim(am_profits_fd.index[0],am_profits_fd.index[-1])
plt.tight_layout()
plt.show()
sys.exit()


"""
#
# Graph 2
# Histogram of trade returns (USD) versus normal distribution
#

import scipy.stats as stats
h = am_trades.loc[:,'Profit'].tolist()
h.sort()
fit = stats.norm.pdf(h, np.mean(h), np.std(h))
plt.plot(h, fit,'-o', color=sns.xkcd_rgb["medium blue"])
plt.hist(am_trades.loc[:,'Profit'], density=1, bins=25, color=sns.xkcd_rgb["black"])
plt.show()


"""

#
# Graph 3
# Scatter plot of profit versus trade duration, with datapoints coloured by close code
#
plt.tick_params(axis='both', which='major', labelsize=14)
close_code=pd.DataFrame()
for i in range(0, am_trades.shape[0]):
	x=am_trades.loc[i,'Close code']
	close_code.set_value(i, x, am_trades.loc[i, 'Profit'])
	close_code.set_value(i, str(x)+" Dur", am_trades.loc[i, 'Duration'])
	
plt.scatter(close_code['3 Dur'], close_code[3], c=sns.xkcd_rgb["black"], label="End of Window")
plt.scatter(close_code['2 Dur'], close_code[2], c=sns.xkcd_rgb["red orange"], label="Stop Loss")
plt.scatter(close_code['1 Dur'], close_code[1], c=sns.xkcd_rgb["medium blue"], label="Convergence")
plt.legend(fontsize=18)
plt.xlim(0,500)
plt.xlabel("Trade Duration", size=24)
plt.ylabel("Trade Profit", size=24)
	
#plt.hist(am_gr_fd.iloc[:,1:].mean(1), density=1, bins=50, range=(am_gr_fd.iloc[:,1:].mean(1).min(), am_gr_fd.iloc[:,1:].mean(1).max()))
#plt.hist(am_gr_fd.iloc[:,1:].mean(1), density=1, bins=100, range=(-0.5, 0.5))
#plt.hist(am_gr_fd[maxcol], density=1, bins=10, range=(-1, 1))
#plt.hist(am_gr_fd[mincol], density=1, bins=10, range=(-1, 1))
plt.show()

"""
plt.hist(am_profits_fd.iloc[:,1:].mean(1), density=1, bins=100, range=(-0.5, 0.5))
plt.show()
"""



#
# Graph 4 (broken)
# Histogram of all strategies FD return series
#

"""
l=0
hugedict={}
for i in range(1, am_profits_fd.shape[1]):
	for j in range(0, am_profits_fd.shape[0]):
		hugedict[l]=am_profits_fd.iloc[j,i]
		l+=1
		
print len(hugedict)
hugedf=pd.DataFrame.from_dict(data=hugedict, orient='index')
print hugedf.shape
hugedf=hugedf.dropna()
print hugedf.shape

# 1. Basic histograms
plt.hist(hugedf, density=1)
N, bins, patches = plt.hist(hugedf, density=1)
plt.show()

# Colouring bins by their frequency
fracs = N / N.max()
norm = colors.Normalize(fracs.min(), fracs.max())
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
	
plt.show()

mu = np.mean(x)
sigma = np.std(x)
z = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
plt.plot(bins, z, '--', color = (0.8, 0.1, 0.1))
	
plt.xlabel(str(data_df.columns.values[i]))
plt.ylabel("Density")	
title = "Distribution of Daily Change in Portfolio Value (USD)"
plt.title(title)
figure = plt.gcf()
pp.savefig(figure)
plt.show()"""




#
# Graph 5
# Distribution of trades, profit, CoEE across pairs
#

# Note: These graphs make more sense for individual strategies rather than entire models due to likely double counting effects

d = Counter(am_trades['Pair'])
#pairs = Counter(am_trades['Pair'])
counted = pd.DataFrame.from_dict(d, orient='index').reset_index()
counted.columns = ['Pair', 'Count']

counted.sort_values('Count', inplace=True, ascending=False)
counted.reset_index(inplace=True)
counted.drop('index', axis=1, inplace=True)
plt.barh(counted.index, counted["Count"], tick_label=counted["Pair"])
plt.yticks(fontsize=12)
plt.ylim([-1,155])
plt.show()

# Trying to add a column of profit to the counted DF
for i in range(0, am_trades.shape[0]):
	pair = am_trades.loc[i,'Pair']
	profit = am_trades.loc[i,'Profit']
	dex = counted.index[counted['Pair'] == pair].tolist()
	if 'Profit' in counted.columns:
		previous = counted.loc[dex, 'Profit']
		profit = profit + profit
	if profit != 0:
		counted.set_value(dex, 'Profit', profit)
		

counted.sort_values('Profit', inplace=True, ascending=False)
counted.reset_index(inplace=True)
counted.drop('index', axis=1, inplace=True)
plt.barh(counted.index, counted["Profit"], tick_label=counted["Pair"])
plt.yticks(fontsize=12)
plt.ylim([-1,155])
plt.show()

for i in range(0, counted.shape[0]):
	profit = counted.loc[i, 'Profit']
	trades = counted.loc[i, 'Count']
	coee = profit / trades*2
	coee = coee*100
	counted.loc[i,'CoEE']=coee 


counted.sort_values('CoEE', inplace=True)
counted.reset_index(inplace=True)
counted.drop('index', axis=1, inplace=True)

plt.barh(counted.index, counted["CoEE"], tick_label=counted["Pair"])
plt.yticks(fontsize=12)
plt.ylim([-1,155])
plt.show()

# Counted needs to be sorted before being fed into this
counted.sort_values('CoEE', inplace=True)

# Shift everything so that it is not negative
shift=counted.loc[0, 'CoEE']
counted['Shifty McShift']=counted['CoEE']+abs(shift)
total = counted['Shifty McShift'].sum()

percentiles=pd.DataFrame(data=None)

for i in range(0, counted.shape[0]):
	percentiles.loc[i,'Equality Percentiles']=float(i)/float(counted.shape[0])
	percentiles.loc[i,'CoEE Percentiles']=counted.loc[0:i,'Shifty McShift'].sum()/total
	if i == counted.shape[0]-1:
		percentiles.loc[i,'CoEE Percentiles'] = 1
		percentiles.loc[i,'Equality Percentiles'] = 1
	if i == 0:
		percentiles.loc[i,'CoEE Percentiles'] = 0
		percentiles.loc[i,'Equality Percentiles'] = 0

counted.sort_values('Profit', inplace=True)
counted.reset_index(inplace=True)
counted.drop('index', axis=1, inplace=True)

shift=counted.loc[0, 'Profit']
counted['Shifty McShift']=counted['Profit']+abs(shift)
total = counted['Shifty McShift'].sum()

for i in range(0, counted.shape[0]):
	percentiles.loc[i,'Equality Percentiles']=float(i)/float(counted.shape[0])
	percentiles.loc[i,'Profit Percentiles']=counted.loc[0:i,'Shifty McShift'].sum()/total
	if i == counted.shape[0]-1:
		percentiles.loc[i,'Profit Percentiles'] = 1
		percentiles.loc[i,'Equality Percentiles'] = 1
	if i == 0:
		percentiles.loc[i,'Profit Percentiles'] = 0
		percentiles.loc[i,'Equality Percentiles'] = 0
		

counted.sort_values('Count', inplace=True)
counted.reset_index(inplace=True)
counted.drop('index', axis=1, inplace=True)

total = counted['Count'].sum()

for i in range(0, counted.shape[0]):
	percentiles.loc[i,'Equality Percentiles']=float(i)/float(counted.shape[0])
	percentiles.loc[i,'Trade Percentiles']=float(counted.loc[0:i,'Count'].sum())/float(total)
	if i == counted.shape[0]-1:
		percentiles.loc[i,'Trade Percentiles'] = 1
		percentiles.loc[i,'Equality Percentiles'] = 1
	if i == 0:
		percentiles.loc[i,'Trade Percentiles'] = 0
		percentiles.loc[i,'Equality Percentiles'] = 0

plt.rcParams["figure.figsize"] = (15,15)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(percentiles['Equality Percentiles'].sort_values(), percentiles['Equality Percentiles'].sort_values(), c=sns.xkcd_rgb["black"], linestyle="dotted")
plt.plot(percentiles['Equality Percentiles'].sort_values(), percentiles['CoEE Percentiles'].sort_values(), label = 'CoEE Percentiles', c=sns.xkcd_rgb["greenish"])
plt.plot(percentiles['Equality Percentiles'].sort_values(), percentiles['Profit Percentiles'].sort_values(), label = 'Profit Percentiles', c=sns.xkcd_rgb["medium blue"])		
plt.plot(percentiles['Equality Percentiles'].sort_values(), percentiles['Trade Percentiles'].sort_values(), label = 'Trade Percentiles', c=sns.xkcd_rgb["red orange"])		
plt.ylim([0,1])
plt.xlim([0,1])
major_ticks = np.arange(0, 1.01, 0.1)
ax = plt.axes()
ax.set_xticks(major_ticks)
ax.set_yticks(major_ticks)
#plt.ylabel('Percentiles of Trades, Profit and Return on Capital Employed')
plt.xlabel('Percentiles of Traded Pairs', size=24)
plt.ylabel('Percentiles of Metric', size=24)
plt.legend(fontsize=18)
plt.show()

"""
am_trades.sort_values('Profit', inplace=True)
am_trades.reset_index(inplace=True)
am_trades.drop('index', axis=1, inplace=True)
shift=abs(am_trades.loc[0, 'Profit'])
am_trades['Shifted Profit']=am_trades['Profit']+shift

percentiles_2 = pd.DataFrame()
for i in range(0, am_trades.shape[0]):
	percentiles_2.loc[i, 'Equality Percentiles'] = float(i)/float(am_trades.shape[0])
	percentiles_2.loc[i, 'Profit Percentiles'] = am_trades.loc[0:i,'Shifted Profit'].sum()/am_trades['Shifted Profit'].sum()
	if i == 0:
		percentiles_2.loc[i, 'Equality Percentiles'] = 0
	if i == am_trades.shape[0]-1:
		percentiles_2.loc[i, 'Equality Percentiles'] = 1

#print percentiles_2


plt.plot(percentiles_2['Equality Percentiles'].sort_values(), percentiles_2['Equality Percentiles'].sort_values(), c=sns.xkcd_rgb["black"], linestyle="dotted")
plt.plot(percentiles_2['Equality Percentiles'].sort_values(), percentiles_2['Profit Percentiles'].sort_values(), label = 'Profit Percentiles')
plt.ylim([0,1])
plt.xlim([0,1])
major_ticks = np.arange(0, 1.01, 0.1)
ax = plt.axes()
ax.set_xticks(major_ticks)
ax.set_yticks(major_ticks)
#plt.ylabel('Percentiles of Trades, Profit and Return on Capital Employed')
plt.xlabel('Percentiles of Traded Pairs')
plt.legend()
plt.show()
"""
sys.exit()