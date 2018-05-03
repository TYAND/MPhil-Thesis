# -*- coding: utf-8 -*-
"""
Created on Fri Apr 06 14:59:29 2018

@author: Tim
"""

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
plt.rcParams["figure.figsize"] = (16,12)

start=time.clock()
HHMM = time.strftime("%H%M")
HHMMSS = time.strftime("%H:%M:%S")

print
print "Time: %s" %(time.strftime("%H:%M:%S"))
print
print
print "Importing modules."

# Used for ADF testing, EG cointegration testing and running regressions
import statsmodels.tsa.stattools as ts

# Used to generate tuples for all possible unique combinations of the input
import itertools

# Progress bar
from tqdm import tqdm

# Text to speech
from tts_watson.TtsWatson import TtsWatson
ttsWatson = TtsWatson('21be4c9c-5a98-4acd-996d-e9bea794536e','FSQsw3dkvYul', 'en-US_AllisonVoice')

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from collections import Counter

print "All modules imported successfully."
input_filename = 'C:\Users\Asus\Desktop\Output Data\Results 1152\Results 1152 (model 19 trade details).csv'
data_df = pd.read_csv(input_filename)

df = pd.DataFrame(data=2, index=[1,2,3], columns=['A','B','C'])
print df
df.loc[2,'A'] = 4
print df

sys.exit()
name="Analysis by Pair"
writer = pd.ExcelWriter(name + '.xlsx', engine='xlsxwriter')
by_pair_df.to_excel(writer, 'Analysis by Pair')

#plt.barh(counted_df.index[0:10], counted_df.iloc[0:10])
#plt.show()