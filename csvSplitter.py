# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:52:44 2019

@author: brandon
"""

import csv
import re
import pandas as pd
import numpy as np
import shutil
import sys
import string
import glob
import os

#change to your desired path
filepath = "C:/Users/brand/.spyder-py3/all-the-news"
filelist = os.listdir(filepath)
files = ""

for name in filelist:
    files += " " + name

name_part = re.findall(r'articles[1][.]csv', files)
batch_size = 100

for file in name_part:
    df = pd.read_csv(filepath + "/" + file)
    df_length = len(df.index)
    for i in range(df_length):
        begin = i * batch_size
        end = (i + 1) * batch_size
        if (end > df_length):
            end = df_length - 1
        df_temp = df[begin : end]
        df_temp.to_csv((filepath + "/split/articles" + str(i)) + ".csv", header=None, index=None)