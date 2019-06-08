
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
#from stanfordnlp.server import CoreNLPClient
from stanfordcorenlp import StanfordCoreNLP

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


# Get only the parts of the corpus. For now just only process files with 1
name_part = re.findall(r'articles[1][.]csv', files)


numFiles = 0
numSentences = 0
numArticles = 0

entity_index = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
columns = ["entity", "type", "publisher", "author", "article", "line"]
rows = list()

# set up the client
nlp = StanfordCoreNLP(r'C:\stanford-corenlp-full-2018-10-05')

ner_types = {"LOCATION", "STATE_OR_PROVINCE", "CITY", "COUNTRY", 
             "ORGANIZATION", "PERSON", "IDEOLOGY", "TITLE"}

for file in name_part:
    # Read the log file
    try:
        df = pd.read_csv(filePath + "/" + file)
    except SyntaxError:
        print("Syntax error on file: ", file)
    print(filePath + "/" + file)
    print("opening files")
    # Get the number of lines in the dataframe
    print("Number of articles in file: %d" % (df.shape[0]))
    records = df.shape[0]
    numFiles+=1
    for record in range(0, records):
        numArticles+=1        
        if(numArticles > 50):
            break
        publisher = df.at[record, 
                          "publication"]
        article = df.at[record, "title"]
        content = df.at[record, "content"]
        author = df.at[record, "author"]
        lines = content.strip().split(".")
        print("Processing article: %d %s" % (numArticles, article))
        line_number = 0
        for line in lines:
            line_number+=1
            entities = nlp.ner(line)
            for entity in entities:
                if entity[1] in ner_types:
                    rows.append([entity[0], entity[1], 
                                publisher, author, article, 
                                line_number])
                    entity_index[(entity[0], entity[1], publisher)].append([author, article, line_number])
                
concordance = pd.DataFrame(rows, columns=columns)
print("Total entities: %d" % concordance.shape[0])
nlp.close()
print(concordance) 
file_name = filePath + "/concordance.csv"
concordance.to_csv(path_or_buf=file_name, header=None, index=None)
 