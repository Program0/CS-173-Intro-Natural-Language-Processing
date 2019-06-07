
# For printing from pyspark
from __future__ import print_function

import shutil
import sys
import string
import glob
import os
from collections import defaultdict
from collections import Counter
from operator import itemgetter
import numpy as np
import pandas as pd
import csv
import re
from stanfordnlp.server import CoreNLPClient

# Create column names for the dataframe
data_names = ["id", "title", "publication",
              "author", "date", "year", "month", 
              "ulr", "content"]

filePath = "C:/Users/super0/Downloads/all-the-news"

files = ""

filelist = os.listdir(filePath)

# Get all the file names in the brown corpus path
for name in filelist:
    files += " " + name


# Get only the parts of the corpus
name_part = re.findall(r'articles[0-9][.]csv', files)


numFiles = 0
numSentences = 0

#article = defaultdict()
print("Hello world")
print(filePath)
print(files)

# set up the client
with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'], timeout=30000, memory='4G') as client:

    for file in name_part:
        # Read the log file
        df = pd.read_csv(filePath + "/" + file)
        print(filePath + "/" + file)
        print("opening files")
        # Get the number of lines in the dataframe
        print("Number of records in dataframe: %d" % (df.shape[0]))
        print(df.iloc[0])
        content = df.at[0, "content"]
        
        lines = content.strip().split(".")
        text = lines[0]
        print(text)
        ann = client.annotate(text)

        # get the first sentence
        sentence = ann.sentence[0]
        
        # get the first token of the first sentence
        print('---')
        print('first token of first sentence')
        token = sentence.token[0]
        print(token)

        # get the part-of-speech tag
        print('---')
        print('part of speech tag of token')
        token.pos
        print(token.pos)

        # get the named entity tag
        print('---')
        print('named entity tag of token')
        print(token.ner)

        # get an entity mention from the first sentence
        print('---')
        print('first entity mention in sentence')
        print(sentence.mentions[0])

        # access the coref chain
        print('---')
        print('coref chains for the example')
        print(ann.corefChain)

        # Use tokensregex patterns to find who wrote a sentence.
        pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
        matches = client.tokensregex(text, pattern)
        # sentences contains a list with matches for each sentence.
        assert len(matches["sentences"]) == 3
        # length tells you whether or not there are any matches in this
        assert matches["sentences"][1]["length"] == 1
        # You can access matches like most regex groups.
        matches["sentences"][1]["0"]["text"] == "Chris wrote a simple sentence"
        matches["sentences"][1]["0"]["1"]["text"] == "Chris"

        # Use semgrex patterns to directly find who wrote what.
        pattern = '{word:wrote} >nsubj {}=subject >dobj {}=object'
        matches = client.semgrex(text, pattern)
        # sentences contains a list with matches for each sentence.
        assert len(matches["sentences"]) == 3
        # length tells you whether or not there are any matches in this
        assert matches["sentences"][1]["length"] == 1
        # You can access matches like most regex groups.
        matches["sentences"][1]["0"]["text"] == "wrote"
        matches["sentences"][1]["0"]["$subject"]["text"] == "Chris"
        matches["sentences"][1]["0"]["$object"]["text"] == "sentence"

