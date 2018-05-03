# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 12:51:35 2018

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
#210 x 297
plt.rcParams["figure.figsize"] = (40,34)
#plt.rcParams["figure.figsize"] = (12,12)
#plt.rcParams["figure.figsize"] = (18,30) # Seems optimal for full-page big boi graphs
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set()
sns.set_style('ticks')
sns.despine()

import matplotlib.gridspec as gridspec

voltrad = pd.read_csv("C:\Users\Asus\Desktop\Explaining 2011 A.csv", index_col=0)
pairtrad = pd.read_csv("C:\Users\Asus\Desktop\Explaining 2011 B.csv", index_col=0)

plt.tick_params(axis='both', which='major', labelsize=14)

gs1 = plt.GridSpec(8, 1)
gs1.update(left=0.05, right=0.48, wspace=0.05, hspace=0.4, bottom=0.185)
#gs2=grid.GridSpecFromSubplotSpec(4, 1, vspace=0)
#fig, axes = plt.subplots(2,1, sharex=True)

plt.subplot(gs1[0:2, :]).plot(voltrad.iloc[:,0], color=sns.xkcd_rgb["medium blue"], label="Actual Value")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(np.arange(1971, 2018, 2), rotation=90)
plt.xlim([1971,2018])
plt.subplot(gs1[0:2, :]).set_ylabel("FX Volatility", size=18)
plt.subplot(gs1[0:2, :]).axhline(y=voltrad.iloc[:,0].mean(), c=sns.xkcd_rgb["medium blue"], linestyle='--', alpha=0.5, label="1971-2018 Mean")
plt.subplot(gs1[0:2, :]).axvline(x=2011, c=sns.xkcd_rgb["black"], linestyle='--', alpha=0.5)
plt.subplot(gs1[0:2, :]).axhline(y=0.264792057, c=sns.xkcd_rgb["black"], linestyle='--', alpha=0.5, label="2011 Value")
plt.subplot(gs1[0:2, :]).legend(loc=2, fontsize=16)

#gs1 = plt.GridSpec(5, 2, wspace=0, hspace=0.15, height_ratios=[2, 2, 2, 1, 1])
plt.subplot(gs1[2:4, :]).plot(voltrad.iloc[:,1], color=sns.xkcd_rgb["red orange"])
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(np.arange(1971, 2018, 2), rotation=90)
plt.xlim([1971,2018])
plt.subplot(gs1[2:4, :]).axhline(y=voltrad.iloc[:,1].mean(), c=sns.xkcd_rgb["red orange"], linestyle='--', alpha=0.5, label="1971-2018 Mean")
plt.subplot(gs1[2:4, :]).set_ylabel("Total Number of Trades", size=18)
plt.subplot(gs1[2:4, :]).axvline(x=2011, c=sns.xkcd_rgb["black"], linestyle='--', alpha=0.5)
plt.subplot(gs1[2:4, :]).axhline(y=288, c=sns.xkcd_rgb["black"], linestyle='--', alpha=0.5, label="2011 Value")
#plt.legend()


#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()
#grid = plt.GridSpec(2, 2, wspace=0, hspace=0.15)
#plt.subplots_adjust(wspace=0)
#plt.subplot(grid[3:5, :]).subplots_adjust(wspace=0)

x=np.arange(1,30,1)
barlst=plt.subplot(gs1[4:6, :]).bar(x, pairtrad.iloc[:,0].sort_values(ascending=False),tick_label=pairtrad.sort_values('Profit', ascending=False).index, color=sns.xkcd_rgb["medium blue"], alpha=0.75)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.subplot(gs1[4:6, :]).set_ylabel("Total Profit", size=18)
for i in range(-11,-0):
	barlst[i].set_color(sns.xkcd_rgb["red orange"])
#axes[0].barh(x, pairtrad.iloc[:,2],tick_label=pairtrad.index, color=sns.xkcd_rgb["black"], alpha=0.25, label="Duration")
#plt.show()
#plt.ylim([-3000,3000])
plt.xlim([0.5,29.5])
plt.xticks(rotation=75)

#gs2 = plt.GridSpec(5, 2, wspace=0.0, hspace=.5)
#gs2.update(left=0.05, right=0.48, wspace=0.0, hspace=0)
gs2 = plt.GridSpec(8, 1, wspace=0.0, hspace=.0)
gs2.update(left=0.05, right=0.48, wspace=0.0, hspace=0)

plt.subplot(gs2[6:8, :]).sharex=True
plt.subplot(gs2[7, :]).bar(x, -pairtrad.iloc[:,1].sort_values(ascending=False),tick_label=pairtrad.sort_values('Duration', ascending=False).index, color=sns.xkcd_rgb["black"], alpha=0.75, label="Total Number of Trades")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.subplot(gs2[7, :]).set_ylabel("Total Number of Trades", size=18)
plt.xlim([0.5,29.5])
plt.xticks(rotation=75)

#gs3 = plt.GridSpec(5, 2, wspace=0.0, hspace=.0)
#gs3.update(left=0.05, right=0.48, wspace=0.0, hspace=0)
#plt.subplots_adjust(wspace=0, hspace=0)

plt.subplot(gs2[6, :]).bar(x, pairtrad.iloc[:,2].sort_values(ascending=False), tick_label=[""], color=sns.xkcd_rgb["medium blue"], alpha=0.75, label="Total Trade Duration")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.subplot(gs2[6, :]).set_ylabel("Total Duration", size=18)
plt.xlim([0.5,29.5])
#plt.tight_layout()

plt.subplots_adjust(wspace=0, hspace=0)
#plt.tick_params(axis='both', which='minor', labelsize=8)

plt.subplot(gs2[6, :]).get_yaxis().set_label_coords(-0.05,0.5)
plt.subplot(gs2[7, :]).get_yaxis().set_label_coords(-0.05,0.5)
plt.subplot(gs1[0:2, :]).get_yaxis().set_label_coords(-0.05,0.5)
plt.subplot(gs1[2:4, :]).get_yaxis().set_label_coords(-0.05,0.5)
plt.subplot(gs1[4:6, :]).get_yaxis().set_label_coords(-0.05,0.5)

plt.show()