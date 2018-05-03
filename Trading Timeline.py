# -*- coding: utf-8 -*-
"""
Created on Sun Apr 08 17:05:02 2018

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
plt.rcParams["figure.figsize"] = (24,26)
#plt.rcParams["figure.figsize"] = (6,6)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set()
sns.set_style('ticks')
sns.despine()
current_palette = sns.color_palette()
sns.set_palette(current_palette)

begin = np.array([0,250,250,500, 500, 750, 750, 1000, 1000, 1250, 1250])
end =   np.array([249,499,499,749, 749, 999, 999, 1249, 1249, 1499])
event = ["1. Analyse", "2.    Trade", "3. Analyse", "4.    Trade", "5. Analyse", "6.    Trade", "7. Analyse", "8.    Trade", "9. Analyse", "10.    Trade"]
#plt.hplot()
#palette = itertools.cycle(sns.color_palette())
"""for i in range(0,len(begin)):
	if i in [0,2,4]:
		plt.barh(i,  end[i]-begin[i], left=begin[i], color=sns.xkcd_rgb["light red"])
	if i in [1,3,5]:
		plt.barh(i,  end[i]-begin[i], left=begin[i], color=sns.xkcd_rgb["medium blue"])"""

for i in range(0,len(begin)):
	if i in [0,2,4,6, 8]:
		plt.barh(i,  end[i]-begin[i], left=begin[i], color=sns.xkcd_rgb["grey"])
	if i in [1,3,5,7, 9]:
		plt.barh(i,  end[i]-begin[i], left=begin[i], color=sns.xkcd_rgb["medium blue"])
plt.xlim(0,1499)
#plt.xticks(np.arange(0, 1000, 250))
plt.xlabel('Observations', size=28)
plt.yticks(range(len(begin)), event, size=28)
plt.tick_params(axis='x', which='major', labelsize=18)
plt.show()